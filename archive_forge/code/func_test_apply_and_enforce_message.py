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
def test_apply_and_enforce_message():

    def func():
        return pd.DataFrame(columns=['A', 'B', 'C'], index=[0])
    meta = pd.DataFrame(columns=['A', 'D'], index=[0])
    with pytest.raises(ValueError, match="Extra: *['B', 'C']"):
        apply_and_enforce(_func=func, _meta=meta)
    with pytest.raises(ValueError, match=re.escape("Missing: ['D']")):
        apply_and_enforce(_func=func, _meta=meta)