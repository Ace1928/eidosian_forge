import os
import sys
import hashlib
import uuid
import warnings
import collections
import weakref
import requests
import numpy as np
from .. import ndarray
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def split_and_load(data, ctx_list, batch_axis=0, even_split=True):
    """Splits an NDArray into `len(ctx_list)` slices along `batch_axis` and loads
    each slice to one context in `ctx_list`.

    Parameters
    ----------
    data : NDArray or ndarray
        A batch of data.
    ctx_list : list of Context
        A list of Contexts.
    batch_axis : int, default 0
        The axis along which to slice.
    even_split : bool, default True
        Whether to force all slices to have the same number of elements.

    Returns
    -------
    list of NDArrays or ndarrays
        Each corresponds to a context in `ctx_list`.
    """
    array_fn = _mx_np.array if is_np_array() else ndarray.array
    if not isinstance(data, ndarray.NDArray):
        data = array_fn(data, ctx=ctx_list[0])
    if len(ctx_list) == 1:
        return [data.as_in_context(ctx_list[0])]
    slices = split_data(data, len(ctx_list), batch_axis, even_split)
    return [i.as_in_context(ctx) for i, ctx in zip(slices, ctx_list)]