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
def apply_dataarray_vfunc(func, *args, signature: _UFuncSignature, join: JoinOptions='inner', exclude_dims=frozenset(), keep_attrs='override') -> tuple[DataArray, ...] | DataArray:
    """Apply a variable level function over DataArray, Variable and/or ndarray
    objects.
    """
    from xarray.core.dataarray import DataArray
    if len(args) > 1:
        args = tuple(deep_align(args, join=join, copy=False, exclude=exclude_dims, raise_on_invalid=False))
    objs = _all_of_type(args, DataArray)
    if keep_attrs == 'drop':
        name = result_name(args)
    else:
        first_obj = _first_of_type(args, DataArray)
        name = first_obj.name
    result_coords, result_indexes = build_output_coords_and_indexes(args, signature, exclude_dims, combine_attrs=keep_attrs)
    data_vars = [getattr(a, 'variable', a) for a in args]
    result_var = func(*data_vars)
    out: tuple[DataArray, ...] | DataArray
    if signature.num_outputs > 1:
        out = tuple((DataArray(variable, coords=coords, indexes=indexes, name=name, fastpath=True) for variable, coords, indexes in zip(result_var, result_coords, result_indexes)))
    else:
        coords, = result_coords
        indexes, = result_indexes
        out = DataArray(result_var, coords=coords, indexes=indexes, name=name, fastpath=True)
    attrs = merge_attrs([x.attrs for x in objs], combine_attrs=keep_attrs)
    if isinstance(out, tuple):
        for da in out:
            da.attrs = attrs
    else:
        out.attrs = attrs
    return out