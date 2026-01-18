from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
from dask.dataframe._compat import check_to_pydatetime_deprecation
from dask.utils import derived_from
def register_dataframe_accessor(name):
    """
    Register a custom accessor on :class:`dask.dataframe.DataFrame`.

    See :func:`pandas.api.extensions.register_dataframe_accessor` for more.
    """
    from dask.dataframe import DataFrame
    return _register_accessor(name, DataFrame)