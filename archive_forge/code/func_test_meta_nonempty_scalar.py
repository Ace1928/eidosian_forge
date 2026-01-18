from __future__ import annotations
import re
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_300, tm
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.utils import (
from dask.local import get_sync
def test_meta_nonempty_scalar():
    meta = meta_nonempty(np.float64(1.0))
    assert isinstance(meta, np.float64)
    x = pd.Timestamp(2000, 1, 1)
    meta = meta_nonempty(x)
    assert meta is x
    x = pd.DatetimeTZDtype(tz='UTC')
    meta = meta_nonempty(x)
    assert meta == pd.Timestamp(1, tz=x.tz, unit=x.unit)