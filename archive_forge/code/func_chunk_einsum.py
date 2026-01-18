from __future__ import annotations
import numpy as np
from dask.array.core import asarray, blockwise, einsum_lookup
from dask.utils import derived_from
def chunk_einsum(*operands, **kwargs):
    subscripts = kwargs.pop('subscripts')
    ncontract_inds = kwargs.pop('ncontract_inds')
    dtype = kwargs.pop('kernel_dtype')
    einsum = einsum_lookup.dispatch(type(operands[0]))
    chunk = einsum(subscripts, *operands, dtype=dtype, **kwargs)
    return chunk.reshape(chunk.shape + (1,) * ncontract_inds)