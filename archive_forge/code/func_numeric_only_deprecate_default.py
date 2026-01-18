from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
def numeric_only_deprecate_default(func):
    """Decorator for methods that should warn when numeric_only is default"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, DataFrameGroupBy):
            numeric_only = kwargs.get('numeric_only', no_default)
            if not PANDAS_GE_150 and numeric_only is False:
                raise NotImplementedError("'numeric_only=False' is not implemented in Dask.")
            if PANDAS_GE_150 and (not PANDAS_GE_200) and (not self._all_numeric()):
                if numeric_only is no_default:
                    warnings.warn('The default value of numeric_only will be changed to False in the future when using dask with pandas 2.0', FutureWarning)
                elif numeric_only is False and funcname(func) in ('sum', 'prod'):
                    warnings.warn(f'Dropping invalid columns is deprecated. In a future version, a TypeError will be raised. Before calling .{funcname(func)}, select only columns which should be valid for the function', FutureWarning)
        return func(self, *args, **kwargs)
    return wrapper