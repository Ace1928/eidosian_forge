import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
def sdd_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, widths, out=None):
    if a.stride(2) != 1 and a.stride(3) != 1:
        a = a.contiguous()
    if b.stride(2) != 1 and b.stride(3) != 1:
        b = b.contiguous()
    if trans_c:
        a, b = (b, a)
        trans_a, trans_b = (not trans_b, not trans_a)
    a_dim = -2 if trans_a else -1
    b_dim = -1 if trans_b else -2
    Ka, Kb = (a.shape[a_dim], b.shape[b_dim])
    if Ka != Kb:
        raise ValueError(f'Inner dimension mismatch (A: {Ka} vs B: {Kb})')
    if out is None:
        c = torch.empty((a.shape[0], lut.shape[0], block, block), dtype=a.dtype, device=a.device)
    else:
        assert out.shape == (a.shape[0], lut.shape[0], block, block)
        c = out
    grid = [c.shape[1], 1, c.shape[0]]
    _sdd_kernel[grid](a, b, c, a.stride(0), a.stride(1), a.stride(3 if trans_a else 2), a.stride(2 if trans_a else 3), b.stride(0), b.stride(1), b.stride(3 if trans_b else 2), b.stride(2 if trans_b else 3), c.stride(0), c.stride(1), c.stride(2), c.stride(3), Ka, 0, lut, TILE_M=block, TILE_N=block, TILE_K=32, BLOCK=block, num_stages=4, num_warps=4)
    return c