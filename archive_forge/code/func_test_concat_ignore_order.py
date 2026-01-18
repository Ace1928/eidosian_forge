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
@pytest.mark.parametrize('ordered', [True, False])
def test_concat_ignore_order(ordered):
    pdf1 = pd.DataFrame({'x': pd.Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c'], ordered=ordered)})
    ddf1 = dd.from_pandas(pdf1, 2)
    pdf2 = pd.DataFrame({'x': pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a'], ordered=ordered)})
    ddf2 = dd.from_pandas(pdf2, 2)
    expected = pd.concat([pdf1, pdf2])
    expected['x'] = expected['x'].astype('category')
    result = dd.concat([ddf1, ddf2], ignore_order=True)
    assert_eq(result, expected)