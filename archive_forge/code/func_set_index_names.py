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
def set_index_names(self, names=None, axis=0):
    if axis == 0:
        self._modin_frame = self._modin_frame.set_index_names(names)
    else:
        self.columns.names = names