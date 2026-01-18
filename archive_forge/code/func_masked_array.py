from __future__ import annotations
from functools import wraps
import numpy as np
from dask.array import chunk
from dask.array.core import asanyarray, blockwise, elemwise, map_blocks
from dask.array.reductions import reduction
from dask.array.routines import _average
from dask.array.routines import nonzero as _nonzero
from dask.base import normalize_token
from dask.utils import derived_from
@derived_from(np.ma)
def masked_array(data, mask=np.ma.nomask, fill_value=None, **kwargs):
    data = asanyarray(data)
    inds = tuple(range(data.ndim))
    arginds = [inds, data, inds]
    if getattr(fill_value, 'shape', ()):
        raise ValueError('non-scalar fill_value not supported')
    kwargs['fill_value'] = fill_value
    if mask is not np.ma.nomask:
        mask = asanyarray(mask)
        if mask.size == 1:
            mask = mask.reshape((1,) * data.ndim)
        elif data.shape != mask.shape:
            raise np.ma.MaskError('Mask and data not compatible: data shape is %s, and mask shape is %s.' % (repr(data.shape), repr(mask.shape)))
        arginds.extend([mask, inds])
    if 'dtype' in kwargs:
        kwargs['masked_dtype'] = kwargs['dtype']
    else:
        kwargs['dtype'] = data.dtype
    return blockwise(_masked_array, *arginds, **kwargs)