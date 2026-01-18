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
def assert_dask_graph(dask, label):
    if hasattr(dask, 'dask'):
        dask = dask.dask
    assert isinstance(dask, Mapping)
    for k in dask:
        if isinstance(k, tuple):
            k = k[0]
        if k.startswith(label):
            return True
    raise AssertionError(f"given dask graph doesn't contain label: {label}")