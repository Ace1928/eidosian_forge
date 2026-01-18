import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@staticmethod
def is_transformation_kernel(agg_func: Any) -> bool:
    """
        Check whether a passed aggregation function is a transformation.

        Transformation means that the result of the function will be broadcasted
        to the frame's original shape.

        Parameters
        ----------
        agg_func : Any

        Returns
        -------
        bool
        """
    return hashable(agg_func) and agg_func in transformation_kernels.union({'nth', 'head', 'tail'})