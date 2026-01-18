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
def test_meta_nonempty_empty_categories():
    for dtype in ['O', 'f8', 'M8[ns]']:
        idx = pd.CategoricalIndex([], pd.Index([], dtype=dtype), ordered=True, name='foo')
        res = meta_nonempty(idx)
        assert type(res) is pd.CategoricalIndex
        assert type(res.categories) is type(idx.categories)
        assert res.ordered == idx.ordered
        assert res.name == idx.name
        s = idx.to_series()
        res = meta_nonempty(s)
        assert res.dtype == 'category'
        assert s.dtype == 'category'
        assert type(res.cat.categories) is type(s.cat.categories)
        assert res.cat.ordered == s.cat.ordered
        assert res.name == s.name