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
@doc_utils.add_one_column_warning
@doc_utils.add_refer_to('Series.dt.freq')
def dt_freq(self):
    """
        Get the time frequency of the underlying time-series data.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing a single value, the frequency of the data.
        """
    return DateTimeDefault.register(pandas.Series.dt.freq)(self)