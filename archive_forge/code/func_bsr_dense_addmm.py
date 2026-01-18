import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def bsr_dense_addmm(input: torch.Tensor, bsr: torch.Tensor, dense: torch.Tensor, *, beta=1, alpha=1, out: Optional[torch.Tensor]=None, skip_checks: bool=False, max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]]=None, meta: Optional[dict]=None):
    f_name = 'bsr_dense_addmm'
    values = bsr.values()
    crow_indices = bsr.crow_indices()
    col_indices = bsr.col_indices()
    batch_ndim = crow_indices.dim() - 1
    M, K = bsr.shape[batch_ndim:batch_ndim + 2]
    blocksize = values.shape[batch_ndim + 1:batch_ndim + 3]
    N = dense.shape[-1]
    original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
    if out is None:
        out = dense.new_empty(original_batch_dims_broadcasted + (M, N))
    if bsr._nnz() == 0 or alpha == 0:
        if beta == 0:
            out.zero_()
        else:
            out.copy_(input)
            if beta != 1:
                out.mul_(beta)
        return out
    if meta is None:
        meta = bsr_dense_addmm_meta(M, K, N, blocksize[0], blocksize[1], beta, alpha, dtype=out.dtype)
    out_backup = out
    crow_indices, col_indices, values, input, dense, out = prepare_inputs(bsr, input, dense, out)
    BM, BK = blocksize
    SPLIT_N = meta.get('SPLIT_N', N // BM)
    BN = N // SPLIT_N
    dense = tile_to_blocksize(dense, (BK, BN))
    input = tile_to_blocksize(input, (BM, BN))
    out_untiled = out
    out = tile_to_blocksize(out, (BM, BN))
    dot_out_dtype = {torch.float16: tl.float32, torch.bfloat16: tl.float32, torch.float32: tl.float64, torch.float64: tl.float64}[out.dtype]
    n_batches = dense.size(0)
    n_block_rows = crow_indices.size(-1) - 1
    n_block_cols = dense.size(-3)
    full_grid = (n_batches, n_block_cols, n_block_rows)
    if max_grid is not None:
        grid_blocks = tuple(max_grid[:3][::-1]) + (None,) * (3 - len(max_grid[:3]))
    else:
        grid_blocks = None
    tensor_dims_map = {values: (0, None, None), crow_indices: (0, None, -1), col_indices: (0, None, None), input: (0, -3, -4), dense: (0, -3, None), out: (0, -3, -4)}
    assert alpha != 0

    def kernel(grid, *sliced_tensors):
        _bsr_strided_addmm_kernel[grid](*ptr_stride_extractor(*sliced_tensors), beta, alpha, beta_is_one=beta == 1, beta_is_nonzero=beta != 0, alpha_is_one=alpha == 1, BLOCKSIZE_ROW=BM, BLOCKSIZE_INNER=BK, BLOCKSIZE_COL=BN, allow_tf32=dot_out_dtype == tl.float32, acc_dtype=dot_out_dtype, **meta)
    launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)
    if out.data_ptr() != out_backup.data_ptr():
        out_backup.copy_(out_untiled.view(out_backup.shape))
    return out_backup