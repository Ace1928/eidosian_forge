import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@classmethod
def get_aggregation_method(cls, how):
    """
        Return `pandas.DataFrameGroupBy` method that implements the passed `how` UDF applying strategy.

        Parameters
        ----------
        how : {"axis_wise", "group_wise", "transform"}
            `how` parameter of the ``BaseQueryCompiler.groupby_agg``.

        Returns
        -------
        callable(pandas.DataFrameGroupBy, callable, *args, **kwargs) -> [pandas.DataFrame | pandas.Series]

        Notes
        -----
        Visit ``BaseQueryCompiler.groupby_agg`` doc-string for more information about `how` parameter.
        """
    return cls._aggregation_methods_dict[how]