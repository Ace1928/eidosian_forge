from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_VERSION, tm
from dask.dataframe.reshape import _get_dummies_dtype_default
from dask.dataframe.utils import assert_eq
def test_get_dummies_categories_order():
    df = pd.DataFrame({'a': [0.0, 0.0, 1.0, 1.0, 0.0], 'b': [1.0, 0.0, 1.0, 0.0, 1.0]})
    ddf = dd.from_pandas(df, npartitions=1)
    ddf = ddf.categorize(columns=['a', 'b'])
    res_p = pd.get_dummies(df.astype('category'))
    res_d = dd.get_dummies(ddf)
    assert_eq(res_d, res_p)