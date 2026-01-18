from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
from dask.dataframe._compat import check_to_pydatetime_deprecation
from dask.utils import derived_from
def str_extractall(series, pat, flags):
    return series.str.extractall(pat, flags=flags)