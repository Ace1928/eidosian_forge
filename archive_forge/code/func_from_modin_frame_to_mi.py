from typing import Iterator, Optional, Tuple
import numpy as np
import pandas
from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, IndexLabel
from pandas.util._decorators import doc
from modin.utils import func_from_deprecated_location, hashable
from_pandas = func_from_deprecated_location(
from_arrow = func_from_deprecated_location(
from_dataframe = func_from_deprecated_location(
from_non_pandas = func_from_deprecated_location(
def from_modin_frame_to_mi(df, sortorder=None, names=None):
    """
    Make a pandas.MultiIndex from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be converted to pandas.MultiIndex.
    sortorder : int, default: None
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : list-like, optional
        If no names are provided, use the column names, or tuple of column
        names if the columns is a MultiIndex. If a sequence, overwrite
        names with the given sequence.

    Returns
    -------
    pandas.MultiIndex
        The pandas.MultiIndex representation of the given DataFrame.
    """
    from .dataframe import DataFrame
    if isinstance(df, DataFrame):
        from modin.error_message import ErrorMessage
        ErrorMessage.default_to_pandas('`MultiIndex.from_frame`')
        df = df._to_pandas()
    return _original_pandas_MultiIndex_from_frame(df, sortorder, names)