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
def apply_dict_of_variables_vfunc(func, *args, signature: _UFuncSignature, join='inner', fill_value=None, on_missing_core_dim: MissingCoreDimOptions='raise'):
    """Apply a variable level function over dicts of DataArray, DataArray,
    Variable and ndarray objects.
    """
    args = tuple((_as_variables_or_variable(arg) for arg in args))
    names = join_dict_keys(args, how=join)
    grouped_by_name = collect_dict_values(args, names, fill_value)
    result_vars = {}
    for name, variable_args in zip(names, grouped_by_name):
        core_dim_present = _check_core_dims(signature, variable_args, name)
        if core_dim_present is True:
            result_vars[name] = func(*variable_args)
        elif on_missing_core_dim == 'raise':
            raise ValueError(core_dim_present)
        elif on_missing_core_dim == 'copy':
            result_vars[name] = variable_args[0]
        elif on_missing_core_dim == 'drop':
            pass
        else:
            raise ValueError(f'Invalid value for `on_missing_core_dim`: {on_missing_core_dim!r}')
    if signature.num_outputs > 1:
        return _unpack_dict_tuples(result_vars, signature.num_outputs)
    else:
        return result_vars