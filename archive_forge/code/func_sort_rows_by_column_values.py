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
def sort_rows_by_column_values(self, columns, ascending=True, **kwargs):
    if kwargs.get('key', None) is not None:
        raise NotImplementedError('Sort with key function')
    ignore_index = kwargs.get('ignore_index', False)
    na_position = kwargs.get('na_position', 'last')
    return self.__constructor__(self._modin_frame.sort_rows(columns, ascending, ignore_index, na_position), self._shape_hint)