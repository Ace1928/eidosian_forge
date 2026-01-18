import torch
from ... import jit
from ... import language as tl
from ... import next_power_of_2
@staticmethod
def make_lut(layout, block, device):
    _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
    sizes = _empty.clone()
    for h in range(layout.shape[0]):
        sizes = torch.cat((sizes, layout[h, :, :].sum(-1)))
    total_sizes = sizes * block
    offsets = torch.zeros_like(sizes)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    columns = layout.nonzero(as_tuple=False)[:, 2]
    header = torch.stack((sizes, offsets), dim=1).view(-1)
    lut = torch.cat((header, columns)).type(torch.int32).to(device)
    return (lut, int(total_sizes.max()))