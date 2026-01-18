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
@_inherit_docstrings(pandas.crosstab, apilink='pandas.crosstab')
@enable_logging
def crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name: str='All', dropna: bool=True, normalize=False) -> DataFrame:
    """
    Compute a simple cross tabulation of two (or more) factors.
    """
    ErrorMessage.default_to_pandas('`crosstab`')
    pandas_crosstab = pandas.crosstab(index, columns, values, rownames, colnames, aggfunc, margins, margins_name, dropna, normalize)
    return DataFrame(pandas_crosstab)