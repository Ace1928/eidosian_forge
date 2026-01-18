import itertools
import json
import sys
from typing import Iterable, Optional, Tuple, List, Sequence, Union
from pkg_resources._vendor.packaging.version import parse as parse_version
import numpy as np
import pyarrow as pa
from ray.air.util.tensor_extensions.utils import (
from ray._private.utils import _get_pyarrow_version
from ray.util.annotations import PublicAPI
@classmethod
def from_numpy(cls, arr: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]) -> 'ArrowVariableShapedTensorArray':
    """
        Convert an ndarray or an iterable of heterogeneous-shaped ndarrays to an array
        of heterogeneous-shaped, homogeneous-typed tensors.

        Args:
            arr: An ndarray or an iterable of heterogeneous-shaped ndarrays.

        Returns:
            An ArrowVariableShapedTensorArray containing len(arr) tensors of
            heterogeneous shape.
        """
    if not isinstance(arr, (list, tuple, np.ndarray)):
        raise ValueError(f'ArrowVariableShapedTensorArray can only be constructed from an ndarray or a list/tuple of ndarrays, but got: {type(arr)}')
    if len(arr) == 0:
        raise ValueError('Creating empty ragged tensor arrays is not supported.')
    shapes, sizes, raveled = ([], [], [])
    ndim = None
    for a in arr:
        a = np.asarray(a)
        if ndim is not None and a.ndim != ndim:
            raise ValueError(f'ArrowVariableShapedTensorArray only supports tensor elements that all have the same number of dimensions, but got tensor elements with dimensions: {ndim}, {a.ndim}')
        ndim = a.ndim
        shapes.append(a.shape)
        sizes.append(a.size)
        a = np.ravel(a, order='C')
        raveled.append(a)
    sizes = np.array(sizes)
    size_offsets = np.cumsum(sizes)
    total_size = size_offsets[-1]
    if all((_is_contiguous_view(curr, prev) for prev, curr in _pairwise(raveled))):
        np_data_buffer = raveled[-1].base
    else:
        np_data_buffer = np.concatenate(raveled)
    dtype = np_data_buffer.dtype
    pa_dtype = pa.from_numpy_dtype(dtype)
    if pa.types.is_string(pa_dtype):
        if dtype.byteorder == '>' or (dtype.byteorder == '=' and sys.byteorder == 'big'):
            raise ValueError(f'Only little-endian string tensors are supported, but got: {dtype}')
        pa_dtype = pa.binary(dtype.itemsize)
    if dtype.type is np.bool_:
        np_data_buffer = np.packbits(np_data_buffer, bitorder='little')
    data_buffer = pa.py_buffer(np_data_buffer)
    value_array = pa.Array.from_buffers(pa_dtype, total_size, [None, data_buffer])
    size_offsets = np.insert(size_offsets, 0, 0)
    offset_array = pa.array(size_offsets)
    data_array = pa.ListArray.from_arrays(offset_array, value_array)
    shape_array = pa.array(shapes)
    storage = pa.StructArray.from_arrays([data_array, shape_array], ['data', 'shape'])
    type_ = ArrowVariableShapedTensorType(pa_dtype, ndim)
    return pa.ExtensionArray.from_storage(type_, storage)