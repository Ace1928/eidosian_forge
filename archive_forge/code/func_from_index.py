from statsmodels.compat.pandas import (
from abc import ABC, abstractmethod
import datetime as dt
from typing import Optional, Union
from collections.abc import Hashable, Sequence
import numpy as np
import pandas as pd
from scipy.linalg import qr
from statsmodels.iolib.summary import d_or_f
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import freq_to_period
@classmethod
def from_index(cls, index: Union[Sequence[Hashable], pd.DatetimeIndex, pd.PeriodIndex]) -> 'Seasonality':
    """
        Construct a seasonality directly from an index using its frequency.

        Parameters
        ----------
        index : {DatetimeIndex, PeriodIndex}
            An index with its frequency (`freq`) set.

        Returns
        -------
        Seasonality
            The initialized Seasonality instance.
        """
    index = cls._index_like(index)
    if isinstance(index, pd.PeriodIndex):
        freq = index.freq
    elif isinstance(index, pd.DatetimeIndex):
        freq = index.freq if index.freq else index.inferred_freq
    else:
        raise TypeError('index must be a DatetimeIndex or PeriodIndex')
    if freq is None:
        raise ValueError('index must have a freq or inferred_freq set')
    period = freq_to_period(freq)
    return cls(period=period)