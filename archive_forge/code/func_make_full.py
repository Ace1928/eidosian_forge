from __future__ import annotations
import contextlib
import operator
import warnings
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import _concat
from dask.dataframe.utils import (
def make_full():
    return pd.DataFrame({'A': pd.Categorical(['a', 'a'])})