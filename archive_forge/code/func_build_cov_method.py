from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@classmethod
def build_cov_method(cls) -> Callable[['PandasQueryCompiler', int, int], 'PandasQueryCompiler']:
    """
        Build a query compiler method computing the covariance matrix.

        Returns
        -------
        callable(qc: PandasQueryCompiler, min_periods: int, ddof: int) -> PandasQueryCompiler
            A callable matching the ``BaseQueryCompiler.cov`` signature and computing the covariance matrix.
        """
    raise NotImplementedError('Computing covariance is not yet implemented.')