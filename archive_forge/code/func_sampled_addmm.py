import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def sampled_addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta=1.0, alpha=1.0, out: Optional[torch.Tensor]=None, skip_checks: bool=False, max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]]=None):
    f_name = 'sampled_addmm'
    check_bsr_layout(f_name, input)
    input_broadcasted = broadcast_batch_dims_bsr(f_name, input, mat1, mat2)
    if not skip_checks:
        check_device(f_name, mat1, input.device)
        check_device(f_name, mat2, input.device)
        if beta != 0.0 and input.dtype is torch.bool:
            check(False, f'{f_name}(): having beta == {beta} not equal to 0.0 with boolean mask is not allowed.')
        if input.dtype is not torch.bool:
            check_dtype(f_name, mat1, input.dtype)
            check_dtype(f_name, mat2, input.dtype)
        else:
            check_dtype(f_name, mat1, mat2.dtype)
        check_mm_compatible_shapes(f_name, mat1, mat2)
        if out is not None:
            check_bsr_layout(f_name, out)
            check_device(f_name, out, mat1.device)
            check_dtype(f_name, out, input.dtype)
            check(out.shape == input_broadcasted.shape and out._nnz() == input._nnz(), f'{f_name}(): Expects `out` to be of shape {input_broadcasted.shape} and with nnz equal to {input_broadcasted._nnz()} but got out.shape = {out.shape} and out.nnz = {out._nnz()}')
    if out is None:
        out = input_broadcasted.to(mat1.dtype, copy=True)
    else:
        out.copy_(input_broadcasted)
    if out.numel() == 0 or out._nnz() == 0:
        return out
    blocksize = out.values().shape[-2:]
    m = mat1.size(-2)
    n = mat2.size(-1)
    k = mat1.size(-1)
    if alpha == 0.0 or k == 0:
        out.values().mul_(beta)
        return out
    out_backup = out
    crow_indices, col_indices, values, mat1, mat2 = prepare_inputs(out, mat1, mat2)
    mat1 = tile_to_blocksize(mat1, (blocksize[0], k))
    mat2 = tile_to_blocksize(mat2, (k, blocksize[1]))
    tile_k = max(*blocksize)
    _run_sampled_addmm_kernel(alpha, beta, beta == 0.0, blocksize, k, tile_k, values, crow_indices, col_indices, mat1, mat2, max_grid)
    if out_backup.values().stride()[-3:] != values.stride()[-3:]:
        out_backup.values().copy_(values.reshape(out_backup.values().shape))
    return out_backup