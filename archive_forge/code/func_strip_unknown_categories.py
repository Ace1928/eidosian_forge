from __future__ import annotations
import math
import re
import sys
import textwrap
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from numbers import Number
from typing import TypeVar, overload
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
import dask
from dask.base import get_scheduler, is_dask_collection
from dask.core import get_deps
from dask.dataframe import (  # noqa: F401 register pandas extension types
from dask.dataframe._compat import PANDAS_GE_150, tm  # noqa: F401
from dask.dataframe.dispatch import (  # noqa : F401
from dask.dataframe.extensions import make_scalar
from dask.typing import NoDefault, no_default
from dask.utils import (
def strip_unknown_categories(x, just_drop_unknown=False):
    """Replace any unknown categoricals with empty categoricals.

    Useful for preventing ``UNKNOWN_CATEGORIES`` from leaking into results.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.copy()
        if isinstance(x, pd.DataFrame):
            cat_mask = x.dtypes == 'category'
            if cat_mask.any():
                cats = cat_mask[cat_mask].index
                for c in cats:
                    if not has_known_categories(x[c]):
                        if just_drop_unknown:
                            x[c].cat.remove_categories(UNKNOWN_CATEGORIES, inplace=True)
                        else:
                            x[c] = x[c].cat.set_categories([])
        elif isinstance(x, pd.Series):
            if isinstance(x.dtype, pd.CategoricalDtype) and (not has_known_categories(x)):
                x = x.cat.set_categories([])
        if isinstance(x.index, pd.CategoricalIndex) and (not has_known_categories(x.index)):
            x.index = x.index.set_categories([])
    elif isinstance(x, pd.CategoricalIndex) and (not has_known_categories(x)):
        x = x.set_categories([])
    return x