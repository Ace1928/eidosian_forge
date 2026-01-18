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
def apply_series_numba(self):
    if self.engine_kwargs.get('parallel', False):
        raise NotImplementedError("Parallel apply is not supported when raw=False and engine='numba'")
    if not self.obj.index.is_unique or not self.columns.is_unique:
        raise NotImplementedError("The index/columns must be unique when raw=False and engine='numba'")
    self.validate_values_for_numba()
    results = self.apply_with_numba()
    return (results, self.result_index)