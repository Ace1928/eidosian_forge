from __future__ import annotations
import os
import pathlib
from time import sleep
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.dataframe._compat import tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq
from dask.layers import DataFrameIOLayer
from dask.utils import dependency_depth, tmpdir, tmpfile
def test_hdf_empty_dataframe(tmp_path):
    pytest.importorskip('tables')
    from dask.dataframe.io.hdf import dont_use_fixed_error_message
    df = pd.DataFrame({'A': [], 'B': []}, index=[])
    df.to_hdf(tmp_path / 'data.h5', format='fixed', key='df', mode='w')
    with pytest.raises(TypeError, match=dont_use_fixed_error_message):
        dd.read_hdf(tmp_path / 'data.h5', 'df')