from __future__ import annotations
import copyreg
import warnings
import pandas as pd
from packaging.version import Version
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
def rebuild_arrowextensionarray(type_, chunks):
    array = pa.chunked_array(chunks)
    return type_(array)