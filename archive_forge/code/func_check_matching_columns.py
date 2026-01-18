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
def check_matching_columns(meta, actual):
    if not np.array_equal(np.nan_to_num(meta.columns), np.nan_to_num(actual.columns)):
        extra = methods.tolist(actual.columns.difference(meta.columns))
        missing = methods.tolist(meta.columns.difference(actual.columns))
        if extra or missing:
            extra_info = f'  Extra:   {extra}\n  Missing: {missing}'
        else:
            extra_info = 'Order of columns does not match'
        raise ValueError(f'The columns in the computed data do not match the columns in the provided metadata\n{extra_info}')