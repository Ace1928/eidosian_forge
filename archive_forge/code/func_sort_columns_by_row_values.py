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
@doc_utils.add_refer_to('DataFrame.sort_values')
def sort_columns_by_row_values(self, rows, ascending=True, **kwargs):
    """
        Reorder the columns based on the lexicographic order of the given rows.

        Parameters
        ----------
        rows : label or list of labels
            The row or rows to sort by.
        ascending : bool, default: True
            Sort in ascending order (True) or descending order (False).
        kind : {"quicksort", "mergesort", "heapsort"}
        na_position : {"first", "last"}
        ignore_index : bool
        key : callable(pandas.Index) -> pandas.Index, optional
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler that contains result of the sort.
        """
    return DataFrameDefault.register(pandas.DataFrame.sort_values)(self, by=rows, axis=1, ascending=ascending, **kwargs)