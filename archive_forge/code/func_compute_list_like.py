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
def compute_list_like(self, op_name: Literal['agg', 'apply'], selected_obj: Series | DataFrame, kwargs: dict[str, Any]) -> tuple[list[Hashable] | Index, list[Any]]:
    """
        Compute agg/apply results for like-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[Hashable] or Index
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python objects.
        """
    func = cast(list[AggFuncTypeBase], self.func)
    obj = self.obj
    results = []
    keys = []
    if selected_obj.ndim == 1:
        for a in func:
            colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
            args = [self.axis, *self.args] if include_axis(op_name, colg) else self.args
            new_res = getattr(colg, op_name)(a, *args, **kwargs)
            results.append(new_res)
            name = com.get_callable_name(a) or a
            keys.append(name)
    else:
        indices = []
        for index, col in enumerate(selected_obj):
            colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
            args = [self.axis, *self.args] if include_axis(op_name, colg) else self.args
            new_res = getattr(colg, op_name)(func, *args, **kwargs)
            results.append(new_res)
            indices.append(index)
        keys = selected_obj.columns.take(indices)
    return (keys, results)