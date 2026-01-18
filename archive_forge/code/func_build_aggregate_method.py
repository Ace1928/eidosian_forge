import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@classmethod
def build_aggregate_method(cls, key):
    """
        Build function for `QueryCompiler.groupby_agg` that can be executed as default-to-pandas.

        Parameters
        ----------
        key : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `key` function is used.

        Returns
        -------
        callable
            Function that executes groupby aggregation.
        """

    def fn(df, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False, **kwargs):
        """Group DataFrame and apply aggregation function to each group."""
        by = cls.validate_by(by)
        grp = cls._call_groupby(df, by, axis=axis, **groupby_kwargs)
        agg_func = cls.get_func(key, **kwargs)
        result = agg_func(grp, *agg_args, **agg_kwargs)
        return result
    return fn