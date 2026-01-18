from __future__ import annotations
from functools import partial
import pandas as pd
from packaging.version import Version
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
def check_pyarrow_string_supported():
    """Make sure we have all the required versions"""
    if not PANDAS_GE_200:
        raise RuntimeError("Using dask's `dataframe.convert-string` configuration option requires `pandas>=2.0` to be installed.")
    if pa is None or Version(pa.__version__) < Version('12.0.0'):
        raise RuntimeError("Using dask's `dataframe.convert-string` configuration option requires `pyarrow>=12` to be installed.")