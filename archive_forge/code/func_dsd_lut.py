import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
def dsd_lut(layout, block, step, trans, device):
    """
    Generates the look-up table for incrementing pointers in the DSD/DDS matmul.
    Example (BLOCK=32, STEP=16)
    [[1, 0, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [1, 0, 1, 0, 0]]

    Then the offsets for A are
     [0 , 16, 32, 48] <- row 0
      \\----/  \\----/
      col=0   col=3
     [64, 80, 96, 112, 128, 144] <- row 1
      \\----/   \\----/  \\------/
       col=1    col=2    col=3
     [160, 176, 192, 208]
    which leads to increments table
    [0, 16, 16, 16, || 64, 16, 16, 16, 16, 16, || 160, 16, 16, 16]

    Because B is dense, the offsets are
    [0, 16, 96, 112] <- row 0
    [32, 48, 64, 80]  <- row 1
    [0, 16, 64, 80]   <- row 2
    """
    sizes = torch.sum(layout, 2 if trans else 1)
    head_id, col_id = torch.ones_like(sizes).nonzero(as_tuple=True)
    sizes = sizes.flatten()
    segments = sizes * step
    if trans:
        nnz = layout.nonzero(as_tuple=False)
    else:
        nnz = layout.transpose(1, 2).nonzero(as_tuple=False)
    num_blocks = nnz.size(0)
    offsets = torch.zeros_like(sizes)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    offsets = torch.min(offsets, (num_blocks - 1) * torch.ones_like(offsets))
    B_idx = nnz[:, 2] * block
    B_incs = B_idx.clone()
    B_incs[1:] -= B_idx[:-1]
    div = block // step
    B_incs = B_incs.view(-1, 1).repeat(1, div)
    B_incs[:, 1:] = step
    B_incs[:, 0] -= (div - 1) * step
    B_incs[offsets[segments > 0], 0] = B_idx[offsets[segments > 0]]
    B_incs = B_incs.view(-1)
    if trans:
        A_idx = torch.arange(num_blocks, device=layout.device)
    else:
        A_idx = torch.tensor([], dtype=torch.int64, device=layout.device)
        current_offset = 0
        for z in range(layout.size(0)):
            layoutw = layout[z, :, :].clone().long()
            msum = layoutw.sum()
            layoutw[layoutw > 0] = 1 + torch.arange(msum, device=layout.device)
            A_idx = torch.cat((A_idx, current_offset + layoutw.T[layoutw.T > 0] - 1))
            current_offset += msum
    A_incs = A_idx * block * block
    A_incs[1:] -= A_idx[:-1] * block * block
    A_incs = A_incs.view(-1, 1).repeat(1, div)
    if trans:
        A_incs[:, 1:] = step
        A_incs[:, 0] -= (div - 1) * step
    else:
        A_incs[:, 1:] = step * block
        A_incs[:, 0] -= (div - 1) * step * block
    A_incs[offsets[segments > 0], 0] = A_idx[offsets[segments > 0]]
    A_incs = A_incs.view(-1)
    width = col_id.size(0)
    offsets = offsets * 2 * div + 4 * width
    segments = segments * div
    header = torch.stack((offsets, segments, col_id, head_id), dim=1).view(-1).contiguous()
    incs = torch.stack((B_incs, A_incs), dim=1).view(-1).contiguous()
    pad = torch.zeros(20, device=incs.device, dtype=incs.dtype)
    incs = torch.cat((incs, pad))
    lut = torch.cat((header, incs))
    lut = lut.type(torch.int32).to(device)
    return (lut, width)