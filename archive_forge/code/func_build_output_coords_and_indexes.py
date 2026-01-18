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
def build_output_coords_and_indexes(args: Iterable[Any], signature: _UFuncSignature, exclude_dims: Set=frozenset(), combine_attrs: CombineAttrsOptions='override') -> tuple[list[dict[Any, Variable]], list[dict[Any, Index]]]:
    """Build output coordinates and indexes for an operation.

    Parameters
    ----------
    args : Iterable
        List of raw operation arguments. Any valid types for xarray operations
        are OK, e.g., scalars, Variable, DataArray, Dataset.
    signature : _UfuncSignature
        Core dimensions signature for the operation.
    exclude_dims : set, optional
        Dimensions excluded from the operation. Coordinates along these
        dimensions are dropped.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "drop"
        A callable or a string indicating how to combine attrs of the objects being
        merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

        If a callable, it must expect a sequence of ``attrs`` dicts and a context object
        as its only parameters.

    Returns
    -------
    Dictionaries of Variable and Index objects with merged coordinates.
    """
    coords_list = _get_coords_list(args)
    if len(coords_list) == 1 and (not exclude_dims):
        unpacked_coords, = coords_list
        merged_vars = dict(unpacked_coords.variables)
        merged_indexes = dict(unpacked_coords.xindexes)
    else:
        merged_vars, merged_indexes = merge_coordinates_without_align(coords_list, exclude_dims=exclude_dims, combine_attrs=combine_attrs)
    output_coords = []
    output_indexes = []
    for output_dims in signature.output_core_dims:
        dropped_dims = signature.all_input_core_dims - set(output_dims)
        if dropped_dims:
            filtered_coords = {k: v for k, v in merged_vars.items() if dropped_dims.isdisjoint(v.dims)}
            filtered_indexes = filter_indexes_from_coords(merged_indexes, set(filtered_coords))
        else:
            filtered_coords = merged_vars
            filtered_indexes = merged_indexes
        output_coords.append(filtered_coords)
        output_indexes.append(filtered_indexes)
    return (output_coords, output_indexes)