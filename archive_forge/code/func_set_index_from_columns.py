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
def set_index_from_columns(self, keys: List[Hashable], drop: bool=True, append: bool=False):
    """
        Create new row labels from a list of columns.

        Parameters
        ----------
        keys : list of hashable
            The list of column names that will become the new index.
        drop : bool, default: True
            Whether or not to drop the columns provided in the `keys` argument.
        append : bool, default: True
            Whether or not to add the columns in `keys` as new levels appended to the
            existing index.

        Returns
        -------
        BaseQueryCompiler
            A new QueryCompiler with updated index.
        """
    return DataFrameDefault.register(pandas.DataFrame.set_index)(self, keys=keys, drop=drop, append=append)