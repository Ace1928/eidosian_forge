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
def cast_function_modin2pandas(func):
    """
    Replace Modin functions with pandas functions if `func` is callable.

    Parameters
    ----------
    func : object

    Returns
    -------
    object
    """
    if callable(func):
        if func.__module__ == 'modin.pandas.series':
            func = getattr(pandas.Series, func.__name__)
        elif func.__module__ in ('modin.pandas.dataframe', 'modin.pandas.base'):
            func = getattr(pandas.DataFrame, func.__name__)
    return func