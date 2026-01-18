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
def apply_list_or_dict_like(self) -> DataFrame | Series:
    """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Result when self.func is a list-like or dict-like, None otherwise.
        """
    if self.engine == 'numba':
        raise NotImplementedError("The 'numba' engine doesn't support list-like/dict likes of callables yet.")
    if self.axis == 1 and isinstance(self.obj, ABCDataFrame):
        return self.obj.T.apply(self.func, 0, args=self.args, **self.kwargs).T
    func = self.func
    kwargs = self.kwargs
    if is_dict_like(func):
        result = self.agg_or_apply_dict_like(op_name='apply')
    else:
        result = self.agg_or_apply_list_like(op_name='apply')
    result = reconstruct_and_relabel_result(result, func, **kwargs)
    return result