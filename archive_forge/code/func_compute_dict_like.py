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
def compute_dict_like(self, op_name: Literal['agg', 'apply'], selected_obj: Series | DataFrame, selection: Hashable | Sequence[Hashable], kwargs: dict[str, Any]) -> tuple[list[Hashable], list[Any]]:
    """
        Compute agg/apply results for dict-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        selection : hashable or sequence of hashables
            Used by GroupBy, Window, and Resample if selection is applied to the object.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[hashable]
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python object.
        """
    from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
    obj = self.obj
    is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
    func = cast(AggFuncTypeDict, self.func)
    func = self.normalize_dictlike_arg(op_name, selected_obj, func)
    is_non_unique_col = selected_obj.ndim == 2 and selected_obj.columns.nunique() < len(selected_obj.columns)
    if selected_obj.ndim == 1:
        colg = obj._gotitem(selection, ndim=1)
        results = [getattr(colg, op_name)(how, **kwargs) for _, how in func.items()]
        keys = list(func.keys())
    elif not is_groupby and is_non_unique_col:
        results = []
        keys = []
        for key, how in func.items():
            indices = selected_obj.columns.get_indexer_for([key])
            labels = selected_obj.columns.take(indices)
            label_to_indices = defaultdict(list)
            for index, label in zip(indices, labels):
                label_to_indices[label].append(index)
            key_data = [getattr(selected_obj._ixs(indice, axis=1), op_name)(how, **kwargs) for label, indices in label_to_indices.items() for indice in indices]
            keys += [key] * len(key_data)
            results += key_data
    else:
        results = [getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs) for key, how in func.items()]
        keys = list(func.keys())
    return (keys, results)