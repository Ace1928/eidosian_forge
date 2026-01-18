from __future__ import annotations
import abc
from collections import defaultdict
import functools
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._config import option_context
from pandas._libs import lib
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
def normalize_keyword_aggregation(kwargs: dict) -> tuple[MutableMapping[Hashable, list[AggFuncTypeBase]], tuple[str, ...], npt.NDArray[np.intp]]:
    """
    Normalize user-provided "named aggregation" kwargs.
    Transforms from the new ``Mapping[str, NamedAgg]`` style kwargs
    to the old Dict[str, List[scalar]]].

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    aggspec : dict
        The transformed kwargs.
    columns : tuple[str, ...]
        The user-provided keys.
    col_idx_order : List[int]
        List of columns indices.

    Examples
    --------
    >>> normalize_keyword_aggregation({"output": ("input", "sum")})
    (defaultdict(<class 'list'>, {'input': ['sum']}), ('output',), array([0]))
    """
    from pandas.core.indexes.base import Index
    aggspec = defaultdict(list)
    order = []
    columns, pairs = list(zip(*kwargs.items()))
    for column, aggfunc in pairs:
        aggspec[column].append(aggfunc)
        order.append((column, com.get_callable_name(aggfunc) or aggfunc))
    uniquified_order = _make_unique_kwarg_list(order)
    aggspec_order = [(column, com.get_callable_name(aggfunc) or aggfunc) for column, aggfuncs in aggspec.items() for aggfunc in aggfuncs]
    uniquified_aggspec = _make_unique_kwarg_list(aggspec_order)
    col_idx_order = Index(uniquified_aggspec).get_indexer(uniquified_order)
    return (aggspec, columns, col_idx_order)