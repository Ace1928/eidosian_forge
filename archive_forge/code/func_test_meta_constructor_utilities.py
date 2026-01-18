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
@pytest.mark.parametrize('data', [pd.DataFrame([0]), pd.Series([0]), pd.Index([0]), dd.from_dict({'x': [0]}, npartitions=1), dd.from_dict({'x': [0]}, npartitions=1).x, dd.from_dict({'x': [0]}, npartitions=1).index])
def test_meta_constructor_utilities(data):
    assert meta_series_constructor(data) is pd.Series
    assert meta_frame_constructor(data) is pd.DataFrame