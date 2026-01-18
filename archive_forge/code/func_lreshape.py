from __future__ import annotations
import warnings
from typing import Hashable, Iterable, Mapping, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import NoDefault, no_default
from pandas._typing import ArrayLike, DtypeBackend, Scalar, npt
from pandas.core.dtypes.common import is_list_like
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import enable_logging
from modin.pandas.io import to_pandas
from modin.utils import _inherit_docstrings
from .base import BasePandasDataset
from .dataframe import DataFrame
from .series import Series
@enable_logging
def lreshape(data: DataFrame, groups, dropna=True) -> DataFrame:
    """
    Reshape wide-format data to long. Generalized inverse of ``DataFrame.pivot``.

    Accepts a dictionary, `groups`, in which each key is a new column name
    and each value is a list of old column names that will be "melted" under
    the new column name as part of the reshape.

    Parameters
    ----------
    data : DataFrame
        The wide-format DataFrame.
    groups : dict
        Dictionary in the form: `{new_name : list_of_columns}`.
    dropna : bool, default: True
        Whether include columns whose entries are all NaN or not.

    Returns
    -------
    DataFrame
        Reshaped DataFrame.
    """
    if not isinstance(data, DataFrame):
        raise ValueError('can not lreshape with instance of type {}'.format(type(data)))
    ErrorMessage.default_to_pandas('`lreshape`')
    return DataFrame(pandas.lreshape(to_pandas(data), groups, dropna=dropna))