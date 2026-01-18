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
@doc_utils.add_refer_to('DataFrame.update')
def df_update(self, other, **kwargs):
    """
        Update values of `self` using non-NA values of `other` at the corresponding positions.

        If axes are not equal, perform frames alignment first.

        Parameters
        ----------
        other : BaseQueryCompiler
            Frame to grab replacement values from.
        join : {"left"}
            Specify type of join to align frames if axes are not equal
            (note: currently only one type of join is implemented).
        overwrite : bool
            Whether to overwrite every corresponding value of self, or only if it's NAN.
        filter_func : callable(pandas.Series, pandas.Series) -> numpy.ndarray<bool>
            Function that takes column of the self and return bool mask for values, that
            should be overwritten in the self frame.
        errors : {"raise", "ignore"}
            If "raise", will raise a ``ValueError`` if `self` and `other` both contain
            non-NA data in the same place.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with updated values.
        """
    return BinaryDefault.register(pandas.DataFrame.update, inplace=True)(self, other=other, **kwargs)