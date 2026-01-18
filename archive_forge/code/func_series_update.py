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
@doc_utils.add_refer_to('Series.update')
def series_update(self, other, **kwargs):
    """
        Update values of `self` using values of `other` at the corresponding indices.

        Parameters
        ----------
        other : BaseQueryCompiler
            One-column query compiler with updated values.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with updated values.
        """
    return BinaryDefault.register(pandas.Series.update, inplace=True)(self, other=other, squeeze_self=True, squeeze_other=True, **kwargs)