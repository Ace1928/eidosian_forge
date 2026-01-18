from functools import wraps
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.common import is_bool_dtype, is_integer_dtype
from modin.core.storage_formats import BaseQueryCompiler
from modin.core.storage_formats.base.query_compiler import (
from modin.core.storage_formats.base.query_compiler import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
def groupby_agg(self, by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, how='axis_wise', drop=False, series_groupby=False):
    if callable(agg_func):
        raise NotImplementedError('Python callable is not a valid aggregation function for HDK storage format.')
    if how != 'axis_wise':
        raise NotImplementedError(f"'{how}' type of groupby-aggregation functions is not supported for HDK storage format.")
    new_frame = self._modin_frame.groupby_agg(by, axis, agg_func, groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop)
    return self.__constructor__(new_frame)