from __future__ import annotations
import contextlib
import functools
import itertools
import math
import numbers
import warnings
import numpy as np
from tlz import concat, frequencies
from dask.array.core import Array
from dask.array.numpy_compat import AxisError
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import has_keyword, is_arraylike, is_cupy_type, typename
def compute_meta(func, _dtype, *args, **kwargs):
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        args_meta = [meta_from_array(x) if is_arraylike(x) else x for x in args]
        kwargs_meta = {k: meta_from_array(v) if is_arraylike(v) else v for k, v in kwargs.items()}
        if isinstance(func, np.vectorize):
            meta = func(*args_meta)
        else:
            try:
                if has_keyword(func, 'computing_meta'):
                    kwargs_meta['computing_meta'] = True
                meta = func(*args_meta, **kwargs_meta)
            except TypeError as e:
                if any((s in str(e) for s in ['unexpected keyword argument', 'is an invalid keyword for', 'Did not understand the following kwargs'])):
                    raise
                else:
                    return None
            except ValueError as e:
                if len(args_meta) == 1 and 'zero-size array to reduction operation' in str(e):
                    meta = args_meta[0]
                else:
                    return None
            except Exception:
                return None
        if _dtype and getattr(meta, 'dtype', None) != _dtype:
            with contextlib.suppress(AttributeError):
                meta = meta.astype(_dtype)
        if np.isscalar(meta):
            meta = np.array(meta)
        return meta