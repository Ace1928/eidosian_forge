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
def test_concat_unknown_divisions():
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([4, 3, 2, 1])
    aa = dd.from_pandas(a, npartitions=2, sort=False).clear_divisions()
    bb = dd.from_pandas(b, npartitions=2, sort=False).clear_divisions()
    assert not aa.known_divisions
    with pytest.warns(UserWarning):
        assert_eq(pd.concat([a, b], axis=1), dd.concat([aa, bb], axis=1))
    cc = dd.from_pandas(b, npartitions=1, sort=False)
    if DASK_EXPR_ENABLED:
        with pytest.raises(ValueError):
            dd.concat([aa, cc], axis=1).optimize()
    else:
        with pytest.raises(ValueError):
            dd.concat([aa, cc], axis=1)
    with warnings.catch_warnings(record=True) as record:
        dd.concat([aa, bb], axis=1, ignore_unknown_divisions=True)
    assert not record