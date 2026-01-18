import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def bsr_dense_mm(bsr: torch.Tensor, dense: torch.Tensor, *, out: Optional[torch.Tensor]=None, skip_checks: bool=False, max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]]=None, meta: Optional[dict]=None, enable_bsr_scatter_mm: bool=True):
    f_name = 'bsr_dense_mm'
    m, kl = bsr.shape[-2:]
    if not skip_checks:
        check_bsr_layout(f_name, bsr)
        check_device(f_name, bsr, dense.device)
        check_dtype(f_name, bsr, dense.dtype)
        check_mm_compatible_shapes(f_name, bsr, dense)
        n = dense.size(-1)
        row_block, col_block = bsr.values().shape[-2:]
        check(not n % row_block, f'bsr_dense_mm(): dense.size(-1) == {n} should be divisible by blocksize[0] == {row_block}.')
        check_blocksize(f_name, (row_block, col_block))
    else:
        kr, n = dense.shape[-2:]
    original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
    if out is not None and (not skip_checks):
        expected_out_shape = original_batch_dims_broadcasted + (m, n)
        check(out.shape == expected_out_shape, f'bsr_dense_mm(): `out` argument has wrong shape, expected {expected_out_shape}, but got {out.shape}.')
        check(out.is_contiguous() or out.transpose(-2, -1).is_contiguous(), 'bsr_dense_mm(): only row-major/col-major `out` arguments are supported, i.e. (out.is_contiguous() or out.transpose(-2, -1).is_contiguous()) should be True.')
    if out is None:
        out = dense.new_empty(original_batch_dims_broadcasted + (m, n))
    if bsr._nnz() == 0:
        return out.zero_()
    blocksize = bsr.values().shape[-2:]
    if enable_bsr_scatter_mm and max(blocksize) == 16 and (bsr.dense_dim() == 0) and (bsr.ndim == 2):
        dtype = bsr.dtype
        if dtype in {torch.float16, torch.bfloat16} and (m >= 4096 and n >= 8192 or (m == 2048 and n >= 32768) or n >= 131072) or (dtype == torch.float32 and (m >= 1024 or (m == 512 and n >= 512) or (m == 256 and n >= 2048))):
            return bsr_scatter_mm(bsr, dense, out=out)
    if meta is None:
        meta = bsr_dense_mm_meta(m, kl, n, blocksize[0], blocksize[1])
    else:
        meta = bsr_dense_mm_meta(m, kl, n, blocksize[0], blocksize[1], **meta)
    out_backup = out
    crow_indices, col_indices, values, dense, out = prepare_inputs(bsr, dense, out)
    dense = tile_to_blocksize(dense, blocksize[::-1])
    out_untiled = out
    out = tile_to_blocksize(out, (blocksize[0], blocksize[0]))
    _run_dense_rowspace_kernel(blocksize, values, crow_indices, col_indices, dense, out, max_grid, meta)
    if out.data_ptr() != out_backup.data_ptr():
        out_backup.copy_(out_untiled.view(out_backup.shape))
    return out_backup