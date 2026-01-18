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
def resample_transform(self, resample_kwargs, arg, *args, **kwargs):
    """
        Resample time-series data and apply aggregation on it.

        Group data into intervals by time-series row/column with
        a specified frequency and call passed function on each group.
        In contrast to ``resample_app_df`` apply function to the whole group,
        instead of a single axis.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.
        arg : callable(pandas.DataFrame) -> pandas.Series
        *args : iterable
            Positional arguments to pass to function.
        **kwargs : dict
            Keyword arguments to pass to function.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the result of passed function.
        """
    return ResampleDefault.register(pandas.core.resample.Resampler.transform)(self, resample_kwargs, arg, *args, **kwargs)