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
def test_check_matching_columns_raises_appropriate_errors():
    df = pd.DataFrame(columns=['a', 'b', 'c'])
    meta = pd.DataFrame(columns=['b', 'a', 'c'])
    with pytest.raises(ValueError, match='Order of columns does not match'):
        assert check_matching_columns(meta, df)
    meta = pd.DataFrame(columns=['a', 'b', 'c', 'd'])
    with pytest.raises(ValueError, match="Missing: \\['d'\\]"):
        assert check_matching_columns(meta, df)
    meta = pd.DataFrame(columns=['a', 'b'])
    with pytest.raises(ValueError, match="Extra:   \\['c'\\]"):
        assert check_matching_columns(meta, df)