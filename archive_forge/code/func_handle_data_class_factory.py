from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def handle_data_class_factory(endog, exog):
    """
    Given inputs
    """
    if data_util._is_using_ndarray_type(endog, exog):
        klass = ModelData
    elif data_util._is_using_pandas(endog, exog):
        klass = PandasData
    elif data_util._is_using_patsy(endog, exog):
        klass = PatsyData
    elif data_util._is_using_ndarray(endog, exog):
        klass = ModelData
    else:
        raise ValueError('unrecognized data structures: %s / %s' % (type(endog), type(exog)))
    return klass