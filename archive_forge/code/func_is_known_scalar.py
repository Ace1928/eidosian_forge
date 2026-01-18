from __future__ import annotations
import numbers
import typing
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from ..exceptions import PlotnineError
def is_known_scalar(value):
    """
    Return True if value is a type we expect in a dataframe
    """

    def _is_datetime_or_timedelta(value):
        return pd.Series(value).dtype.kind in ('M', 'm')
    return not np.iterable(value) and (isinstance(value, numbers.Number) or _is_datetime_or_timedelta(value))