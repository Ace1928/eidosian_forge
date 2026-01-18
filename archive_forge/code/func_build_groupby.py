import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@classmethod
def build_groupby(cls, func):
    """
        Build function that groups DataFrame and applies aggregation function to the every group.

        Parameters
        ----------
        func : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `func` function is used.

        Returns
        -------
        callable
            Function that takes pandas DataFrame and does GroupBy aggregation.
        """
    if cls.is_aggregate(func):
        return cls.build_aggregate_method(func)
    return cls.build_groupby_reduce_method(func)