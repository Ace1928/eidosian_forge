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
def merge_coords(objects: Iterable[CoercibleMapping], compat: CompatOptions='minimal', join: JoinOptions='outer', priority_arg: int | None=None, indexes: Mapping[Any, Index] | None=None, fill_value: object=dtypes.NA) -> tuple[dict[Hashable, Variable], dict[Hashable, Index]]:
    """Merge coordinate variables.

    See merge_core below for argument descriptions. This works similarly to
    merge_core, except everything we don't worry about whether variables are
    coordinates or not.
    """
    _assert_compat_valid(compat)
    coerced = coerce_pandas_values(objects)
    aligned = deep_align(coerced, join=join, copy=False, indexes=indexes, fill_value=fill_value)
    collected = collect_variables_and_indexes(aligned, indexes=indexes)
    prioritized = _get_priority_vars_and_indexes(aligned, priority_arg, compat=compat)
    variables, out_indexes = merge_collected(collected, prioritized, compat=compat)
    return (variables, out_indexes)