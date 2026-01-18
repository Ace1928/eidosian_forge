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
@doc_utils.add_refer_to('GroupBy.rolling')
def groupby_rolling(self, by, agg_func, axis, groupby_kwargs, rolling_kwargs, agg_args, agg_kwargs, drop=False):
    """
        Group QueryCompiler data and apply passed aggregation function to a rolling window in each group.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        agg_func : str, dict or callable(Series | DataFrame) -> scalar | Series | DataFrame
            Function to apply to the GroupBy object.
        axis : {0, 1}
            Axis to group and apply aggregation function along.
            0 is for index, when 1 is for columns.
        groupby_kwargs : dict
            GroupBy parameters as expected by ``modin.pandas.DataFrame.groupby`` signature.
        rolling_kwargs : dict
            Parameters to build a rolling window as expected by ``modin.pandas.window.RollingGroupby`` signature.
        agg_args : list-like
            Positional arguments to pass to the `agg_func`.
        agg_kwargs : dict
            Key arguments to pass to the `agg_func`.
        drop : bool, default: False
            If `by` is a QueryCompiler indicates whether or not by-data came
            from the `self`.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing the result of groupby aggregation.
        """
    if isinstance(agg_func, str):
        str_func = agg_func

        def agg_func(window, *args, **kwargs):
            return getattr(window, str_func)(*args, **kwargs)
    else:
        assert callable(agg_func)
    return self.groupby_agg(by=by, agg_func=lambda grp, *args, **kwargs: agg_func(grp.rolling(**rolling_kwargs), *args, **kwargs), axis=axis, groupby_kwargs=groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, how='direct', drop=drop)