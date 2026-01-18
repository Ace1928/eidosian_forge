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
def getitem_array(self, key):
    if isinstance(key, type(self)):
        new_modin_frame = self._modin_frame.filter(key._modin_frame)
        return self.__constructor__(new_modin_frame, self._shape_hint)
    if is_bool_indexer(key):
        return self.default_to_pandas(lambda df: df[key])
    if any((k not in self.columns for k in key)):
        raise KeyError('{} not index'.format(str([k for k in key if k not in self.columns]).replace(',', '')))
    return self.getitem_column_array(key)