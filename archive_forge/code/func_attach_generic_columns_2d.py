from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def attach_generic_columns_2d(self, result, rownames, colnames=None):
    colnames = colnames or rownames
    rownames = getattr(self, rownames, None)
    colnames = getattr(self, colnames, None)
    return DataFrame(result, index=rownames, columns=colnames)