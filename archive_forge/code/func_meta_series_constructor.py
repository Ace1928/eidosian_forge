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
def meta_series_constructor(like):
    """Return a serial Series constructor

    Parameters
    ----------
    like :
        Any series-like, Index-like or dataframe-like object.
    """
    if is_dask_collection(like):
        try:
            like = like._meta
        except AttributeError:
            raise TypeError(f'{type(like)} not supported by meta_series_constructor')
    if is_dataframe_like(like):
        return like._constructor_sliced
    elif is_series_like(like):
        return like._constructor
    elif is_index_like(like):
        return like.to_frame()._constructor_sliced
    else:
        raise TypeError(f'{type(like)} not supported by meta_series_constructor')