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
def setitem_bool(self, row_loc, col_loc, item):
    """
        Set an item to the given location based on `row_loc` and `col_loc`.

        Parameters
        ----------
        row_loc : BaseQueryCompiler
            Query Compiler holding a Series of booleans.
        col_loc : label
            Column label in `self`.
        item : scalar
            An item to be set.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with the inserted item.

        Notes
        -----
        Currently, this method is only used to set a scalar to the given location.
        """

    def _set_item(df, row_loc, col_loc, item):
        df.loc[row_loc.squeeze(axis=1), col_loc] = item
        return df
    return DataFrameDefault.register(_set_item)(self, row_loc=row_loc, col_loc=col_loc, item=item)