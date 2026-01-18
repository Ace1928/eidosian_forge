import abc
import warnings
from typing import Hashable, List, Optional
import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import DtypeBackend, IndexLabel, Suffixes
from pandas.core.dtypes.common import is_number, is_scalar
from modin.config import StorageFormat
from modin.core.dataframe.algebra.default2pandas import (
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
from . import doc_utils
@doc_utils.doc_str_method(refer_to='pad', params="\n        width : int\n        side : {'left', 'right', 'both'}, default: 'left'\n        fillchar : str, default: ' '")
def str_pad(self, width, side='left', fillchar=' '):
    return StrDefault.register(pandas.Series.str.pad)(self, width, side, fillchar)