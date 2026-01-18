from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.alignment import deep_align
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import (
from xarray.core.utils import Frozen, compat_dict_union, dict_equiv, equivalent
from xarray.core.variable import Variable, as_variable, calculate_dimensions
def merge_core(objects: Iterable[CoercibleMapping], compat: CompatOptions='broadcast_equals', join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='override', priority_arg: int | None=None, explicit_coords: Iterable[Hashable] | None=None, indexes: Mapping[Any, Any] | None=None, fill_value: object=dtypes.NA, skip_align_args: list[int] | None=None) -> _MergeResult:
    """Core logic for merging labeled objects.

    This is not public API.

    Parameters
    ----------
    objects : list of mapping
        All values must be convertible to labeled arrays.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        Compatibility checks to use when merging variables.
    join : {"outer", "inner", "left", "right"}, optional
        How to combine objects with different indexes.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "override"
        How to combine attributes of objects
    priority_arg : int, optional
        Optional argument in `objects` that takes precedence over the others.
    explicit_coords : set, optional
        An explicit list of variables from `objects` that are coordinates.
    indexes : dict, optional
        Dictionary with values given by xarray.Index objects or anything that
        may be cast to pandas.Index objects.
    fill_value : scalar, optional
        Value to use for newly missing values
    skip_align_args : list of int, optional
        Optional arguments in `objects` that are not included in alignment.

    Returns
    -------
    variables : dict
        Dictionary of Variable objects.
    coord_names : set
        Set of coordinate names.
    dims : dict
        Dictionary mapping from dimension names to sizes.
    attrs : dict
        Dictionary of attributes

    Raises
    ------
    MergeError if the merge cannot be done successfully.
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    _assert_compat_valid(compat)
    objects = list(objects)
    if skip_align_args is None:
        skip_align_args = []
    skip_align_objs = [(pos, objects.pop(pos)) for pos in skip_align_args]
    coerced = coerce_pandas_values(objects)
    aligned = deep_align(coerced, join=join, copy=False, indexes=indexes, fill_value=fill_value)
    for pos, obj in skip_align_objs:
        aligned.insert(pos, obj)
    collected = collect_variables_and_indexes(aligned, indexes=indexes)
    prioritized = _get_priority_vars_and_indexes(aligned, priority_arg, compat=compat)
    variables, out_indexes = merge_collected(collected, prioritized, compat=compat, combine_attrs=combine_attrs)
    dims = calculate_dimensions(variables)
    coord_names, noncoord_names = determine_coords(coerced)
    if compat == 'minimal':
        coord_names.intersection_update(variables)
    if explicit_coords is not None:
        coord_names.update(explicit_coords)
    for dim, size in dims.items():
        if dim in variables:
            coord_names.add(dim)
    ambiguous_coords = coord_names.intersection(noncoord_names)
    if ambiguous_coords:
        raise MergeError(f'unable to determine if these variables should be coordinates or not in the merged result: {ambiguous_coords}')
    attrs = merge_attrs([var.attrs for var in coerced if isinstance(var, (Dataset, DataArray))], combine_attrs)
    return _MergeResult(variables, coord_names, dims, out_indexes, attrs)