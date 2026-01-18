from __future__ import annotations
import copyreg
import warnings
import pandas as pd
from packaging.version import Version
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
def reduce_arrowextensionarray(x):
    return (rebuild_arrowextensionarray, (type(x), x._data.combine_chunks()))