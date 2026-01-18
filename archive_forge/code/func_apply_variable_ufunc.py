from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast, overload
import numpy as np
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.alignment import align, deep_align
from xarray.core.common import zeros_like
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.formatting import limit_lines
from xarray.core.indexes import Index, filter_indexes_from_coords
from xarray.core.merge import merge_attrs, merge_coordinates_without_align
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import Dims, T_DataArray
from xarray.core.utils import is_dict_like, is_duck_dask_array, is_scalar, parse_dims
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.util.deprecation_helpers import deprecate_dims
def apply_variable_ufunc(func, *args, signature: _UFuncSignature, exclude_dims=frozenset(), dask='forbidden', output_dtypes=None, vectorize=False, keep_attrs='override', dask_gufunc_kwargs=None) -> Variable | tuple[Variable, ...]:
    """Apply a ndarray level function over Variable and/or ndarray objects."""
    from xarray.core.formatting import short_array_repr
    from xarray.core.variable import Variable, as_compatible_data
    dim_sizes = unified_dim_sizes((a for a in args if hasattr(a, 'dims')), exclude_dims=exclude_dims)
    broadcast_dims = tuple((dim for dim in dim_sizes if dim not in signature.all_core_dims))
    output_dims = [broadcast_dims + out for out in signature.output_core_dims]
    input_data = [broadcast_compat_data(arg, broadcast_dims, core_dims) if isinstance(arg, Variable) else arg for arg, core_dims in zip(args, signature.input_core_dims)]
    if any((is_chunked_array(array) for array in input_data)):
        if dask == 'forbidden':
            raise ValueError('apply_ufunc encountered a chunked array on an argument, but handling for chunked arrays has not been enabled. Either set the ``dask`` argument or load your data into memory first with ``.load()`` or ``.compute()``')
        elif dask == 'parallelized':
            chunkmanager = get_chunked_array_type(*input_data)
            numpy_func = func
            if dask_gufunc_kwargs is None:
                dask_gufunc_kwargs = {}
            else:
                dask_gufunc_kwargs = dask_gufunc_kwargs.copy()
            allow_rechunk = dask_gufunc_kwargs.get('allow_rechunk', None)
            if allow_rechunk is None:
                for n, (data, core_dims) in enumerate(zip(input_data, signature.input_core_dims)):
                    if is_chunked_array(data):
                        for axis, dim in enumerate(core_dims, start=-len(core_dims)):
                            if len(data.chunks[axis]) != 1:
                                raise ValueError(f"dimension {dim} on {n}th function argument to apply_ufunc with dask='parallelized' consists of multiple chunks, but is also a core dimension. To fix, either rechunk into a single array chunk along this dimension, i.e., ``.chunk(dict({dim}=-1))``, or pass ``allow_rechunk=True`` in ``dask_gufunc_kwargs`` but beware that this may significantly increase memory usage.")
                dask_gufunc_kwargs['allow_rechunk'] = True
            output_sizes = dask_gufunc_kwargs.pop('output_sizes', {})
            if output_sizes:
                output_sizes_renamed = {}
                for key, value in output_sizes.items():
                    if key not in signature.all_output_core_dims:
                        raise ValueError(f"dimension '{key}' in 'output_sizes' must correspond to output_core_dims")
                    output_sizes_renamed[signature.dims_map[key]] = value
                dask_gufunc_kwargs['output_sizes'] = output_sizes_renamed
            for key in signature.all_output_core_dims:
                if (key not in signature.all_input_core_dims or key in exclude_dims) and key not in output_sizes:
                    raise ValueError(f"dimension '{key}' in 'output_core_dims' needs corresponding (dim, size) in 'output_sizes'")

            def func(*arrays):
                res = chunkmanager.apply_gufunc(numpy_func, signature.to_gufunc_string(exclude_dims), *arrays, vectorize=vectorize, output_dtypes=output_dtypes, **dask_gufunc_kwargs)
                return res
        elif dask == 'allowed':
            pass
        else:
            raise ValueError(f'unknown setting for chunked array handling in apply_ufunc: {dask}')
    elif vectorize:
        func = _vectorize(func, signature, output_dtypes=output_dtypes, exclude_dims=exclude_dims)
    result_data = func(*input_data)
    if signature.num_outputs == 1:
        result_data = (result_data,)
    elif not isinstance(result_data, tuple) or len(result_data) != signature.num_outputs:
        raise ValueError(f'applied function does not have the number of outputs specified in the ufunc signature. Received a {type(result_data)} with {len(result_data)} elements. Expected a tuple of {signature.num_outputs} elements:\n\n{limit_lines(repr(result_data), limit=10)}')
    objs = _all_of_type(args, Variable)
    attrs = merge_attrs([obj.attrs for obj in objs], combine_attrs=keep_attrs)
    output: list[Variable] = []
    for dims, data in zip(output_dims, result_data):
        data = as_compatible_data(data)
        if data.ndim != len(dims):
            raise ValueError(f'applied function returned data with an unexpected number of dimensions. Received {data.ndim} dimension(s) but expected {len(dims)} dimensions with names {dims!r}, from:\n\n{short_array_repr(data)}')
        var = Variable(dims, data, fastpath=True)
        for dim, new_size in var.sizes.items():
            if dim in dim_sizes and new_size != dim_sizes[dim]:
                raise ValueError(f"size of dimension '{dim}' on inputs was unexpectedly changed by applied function from {dim_sizes[dim]} to {new_size}. Only dimensions specified in ``exclude_dims`` with xarray.apply_ufunc are allowed to change size. The data returned was:\n\n{short_array_repr(data)}")
        var.attrs = attrs
        output.append(var)
    if signature.num_outputs == 1:
        return output[0]
    else:
        return tuple(output)