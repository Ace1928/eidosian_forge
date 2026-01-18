from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core import algorithms
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.io.formats.printing import pprint_thing
def get_grouper(obj: NDFrameT, key=None, axis: Axis=0, level=None, sort: bool=True, observed: bool=False, validate: bool=True, dropna: bool=True) -> tuple[ops.BaseGrouper, frozenset[Hashable], NDFrameT]:
    """
    Create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to axis,level,sort, while
    the passed in axis, level, and sort are 'global'.

    This routine tries to figure out what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    If observed & we have a categorical grouper, only show the observed
    values.

    If validate, then check for key/level overlaps.

    """
    group_axis = obj._get_axis(axis)
    if level is not None:
        if isinstance(group_axis, MultiIndex):
            if is_list_like(level) and len(level) == 1:
                level = level[0]
            if key is None and is_scalar(level):
                key = group_axis.get_level_values(level)
                level = None
        else:
            if is_list_like(level):
                nlevels = len(level)
                if nlevels == 1:
                    level = level[0]
                elif nlevels == 0:
                    raise ValueError('No group keys passed!')
                else:
                    raise ValueError('multiple levels only valid with MultiIndex')
            if isinstance(level, str):
                if obj._get_axis(axis).name != level:
                    raise ValueError(f'level name {level} is not the name of the {obj._get_axis_name(axis)}')
            elif level > 0 or level < -1:
                raise ValueError('level > 0 or level < -1 only valid with MultiIndex')
            level = None
            key = group_axis
    if isinstance(key, Grouper):
        grouper, obj = key._get_grouper(obj, validate=False)
        if key.key is None:
            return (grouper, frozenset(), obj)
        else:
            return (grouper, frozenset({key.key}), obj)
    elif isinstance(key, ops.BaseGrouper):
        return (key, frozenset(), obj)
    if not isinstance(key, list):
        keys = [key]
        match_axis_length = False
    else:
        keys = key
        match_axis_length = len(keys) == len(group_axis)
    any_callable = any((callable(g) or isinstance(g, dict) for g in keys))
    any_groupers = any((isinstance(g, (Grouper, Grouping)) for g in keys))
    any_arraylike = any((isinstance(g, (list, tuple, Series, Index, np.ndarray)) for g in keys))
    if not any_callable and (not any_arraylike) and (not any_groupers) and match_axis_length and (level is None):
        if isinstance(obj, DataFrame):
            all_in_columns_index = all((g in obj.columns or g in obj.index.names for g in keys))
        else:
            assert isinstance(obj, Series)
            all_in_columns_index = all((g in obj.index.names for g in keys))
        if not all_in_columns_index:
            keys = [com.asarray_tuplesafe(keys)]
    if isinstance(level, (tuple, list)):
        if key is None:
            keys = [None] * len(level)
        levels = level
    else:
        levels = [level] * len(keys)
    groupings: list[Grouping] = []
    exclusions: set[Hashable] = set()

    def is_in_axis(key) -> bool:
        if not _is_label_like(key):
            if obj.ndim == 1:
                return False
            items = obj.axes[-1]
            try:
                items.get_loc(key)
            except (KeyError, TypeError, InvalidIndexError):
                return False
        return True

    def is_in_obj(gpr) -> bool:
        if not hasattr(gpr, 'name'):
            return False
        if using_copy_on_write() or warn_copy_on_write():
            try:
                obj_gpr_column = obj[gpr.name]
            except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
                return False
            if isinstance(gpr, Series) and isinstance(obj_gpr_column, Series):
                return gpr._mgr.references_same_values(obj_gpr_column._mgr, 0)
            return False
        try:
            return gpr is obj[gpr.name]
        except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
            return False
    for gpr, level in zip(keys, levels):
        if is_in_obj(gpr):
            in_axis = True
            exclusions.add(gpr.name)
        elif is_in_axis(gpr):
            if obj.ndim != 1 and gpr in obj:
                if validate:
                    obj._check_label_or_level_ambiguity(gpr, axis=axis)
                in_axis, name, gpr = (True, gpr, obj[gpr])
                if gpr.ndim != 1:
                    raise ValueError(f"Grouper for '{name}' not 1-dimensional")
                exclusions.add(name)
            elif obj._is_level_reference(gpr, axis=axis):
                in_axis, level, gpr = (False, gpr, None)
            else:
                raise KeyError(gpr)
        elif isinstance(gpr, Grouper) and gpr.key is not None:
            exclusions.add(gpr.key)
            in_axis = True
        else:
            in_axis = False
        ping = Grouping(group_axis, gpr, obj=obj, level=level, sort=sort, observed=observed, in_axis=in_axis, dropna=dropna) if not isinstance(gpr, Grouping) else gpr
        groupings.append(ping)
    if len(groupings) == 0 and len(obj):
        raise ValueError('No group keys passed!')
    if len(groupings) == 0:
        groupings.append(Grouping(Index([], dtype='int'), np.array([], dtype=np.intp)))
    grouper = ops.BaseGrouper(group_axis, groupings, sort=sort, dropna=dropna)
    return (grouper, frozenset(exclusions), obj)