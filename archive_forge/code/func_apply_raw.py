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
def apply_raw(self, engine='python', engine_kwargs=None):
    """apply to the values as a numpy array"""

    def wrap_function(func):
        """
            Wrap user supplied function to work around numpy issue.

            see https://github.com/numpy/numpy/issues/8352
            """

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, str):
                result = np.array(result, dtype=object)
            return result
        return wrapper
    if engine == 'numba':
        engine_kwargs = {} if engine_kwargs is None else engine_kwargs
        nb_looper = generate_apply_looper(self.func, **engine_kwargs)
        result = nb_looper(self.values, self.axis)
        result = np.squeeze(result)
    else:
        result = np.apply_along_axis(wrap_function(self.func), self.axis, self.values, *self.args, **self.kwargs)
    if result.ndim == 2:
        return self.obj._constructor(result, index=self.index, columns=self.columns)
    else:
        return self.obj._constructor_sliced(result, index=self.agg_axis)