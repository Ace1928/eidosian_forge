from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
from dask.dataframe._compat import check_to_pydatetime_deprecation
from dask.utils import derived_from
def str_cat_none(x):
    if isinstance(x, (Series, Index)):
        x = x.compute()
    return x.str.cat(sep=sep, na_rep=na_rep)