import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def bsr_scatter_mm_indices_data(bsr, other, indices_format='bsr_strided_mm_compressed', **meta_input):
    """Computes indices data for :func:`scatter_mm` used in BSR and
    strided tensor matrix multiplication.
    """
    assert bsr.dense_dim() == 0
    assert bsr.ndim == 2
    crow_indices = bsr.crow_indices()
    col_indices = bsr.col_indices()
    blocksize = bsr.values().shape[-2:]
    M, K = bsr.shape
    Ms, Ks = blocksize
    K_, N = other.shape[-2:]
    assert K_ == K
    nbatches = other.shape[:-2].numel()
    meta = scatter_mm_meta(M, K, N, Ms, Ks, **meta_input)
    if 'allow_tf32' not in meta_input:
        meta.update(allow_tf32=bsr.dtype in {torch.float16, torch.bfloat16})
    SPLIT_N = meta['SPLIT_N']
    indices_data = _bsr_scatter_mm_indices_data(indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, TensorAsKey(bsr))
    if indices_format == 'bsr_strided_mm_compressed':
        meta.update(is_compressed=True)
        return indices_data + (meta,)
    elif indices_format == 'bsr_strided_mm':
        meta.update(is_compressed=False)
        return indices_data + (meta,)
    else:
        return indices_data