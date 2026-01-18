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
def apply_on_series(self, func, *args, **kwargs):
    """
        Apply passed function on underlying Series.

        Parameters
        ----------
        func : callable(pandas.Series) -> scalar, str, list or dict of such
            The function to apply to each row.
        *args : iterable
            Positional arguments to pass to `func`.
        **kwargs : dict
            Keyword arguments to pass to `func`.

        Returns
        -------
        BaseQueryCompiler
        """
    assert self.is_series_like()
    return SeriesDefault.register(pandas.Series.apply)(self, *args, func=func, **kwargs)