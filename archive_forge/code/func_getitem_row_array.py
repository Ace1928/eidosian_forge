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
def getitem_row_array(self, key):
    """
        Get row data for target indices.

        Parameters
        ----------
        key : list-like
            Numeric indices of the rows to pick.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler that contains specified rows.
        """

    def get_row(df, key):
        return df.iloc[key]
    return DataFrameDefault.register(get_row)(self, key=key)