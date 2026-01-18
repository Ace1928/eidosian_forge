from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.dataframe._compat import PANDAS_GE_210, PANDAS_GE_220, IndexingError, tm
from dask.dataframe.indexing import _coerce_loc_index
from dask.dataframe.utils import assert_eq, make_meta, pyarrow_strings_enabled
def test_loc_with_array():
    assert_eq(d.loc[(d.a % 2 == 0).values], full.loc[(full.a % 2 == 0).values])
    if not dd._dask_expr_enabled():
        assert sorted(d.loc[(d.a % 2 == 0).values].dask) == sorted(d.loc[(d.a % 2 == 0).values].dask)
        assert sorted(d.loc[(d.a % 2 == 0).values].dask) != sorted(d.loc[(d.a % 3 == 0).values].dask)