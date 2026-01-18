from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
def test_concat_unknown_divisions_errors():
    a = pd.Series([1, 2, 3, 4, 5, 6])
    b = pd.Series([4, 3, 2, 1])
    aa = dd.from_pandas(a, npartitions=2, sort=False).clear_divisions()
    bb = dd.from_pandas(b, npartitions=2, sort=False).clear_divisions()
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            dd.concat([aa, bb], axis=1).compute()