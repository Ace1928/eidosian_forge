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
def getitem_column_array(self, key, numeric=False, ignore_order=False):
    shape_hint = 'column' if len(key) == 1 else None
    if numeric:
        new_modin_frame = self._modin_frame.take_2d_labels_or_positional(col_positions=key)
    else:
        new_modin_frame = self._modin_frame.take_2d_labels_or_positional(col_labels=key)
    return self.__constructor__(new_modin_frame, shape_hint)