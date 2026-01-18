from __future__ import annotations
import datetime
import typing
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.types import SideOptions
def first_items(self, index: CFTimeIndex):
    """Meant to reproduce the results of the following

        grouper = pandas.Grouper(...)
        first_items = pd.Series(np.arange(len(index)),
                                index).groupby(grouper).first()

        with index being a CFTimeIndex instead of a DatetimeIndex.
        """
    datetime_bins, labels = _get_time_bins(index, self.freq, self.closed, self.label, self.origin, self.offset)
    if self.loffset is not None:
        if not isinstance(self.loffset, (str, datetime.timedelta, BaseCFTimeOffset)):
            raise ValueError(f'`loffset` must be a str or datetime.timedelta object. Got {self.loffset}.')
        if isinstance(self.loffset, datetime.timedelta):
            labels = labels + self.loffset
        else:
            labels = labels + to_offset(self.loffset)
    if index[0] < datetime_bins[0]:
        raise ValueError('Value falls before first bin')
    if index[-1] > datetime_bins[-1]:
        raise ValueError('Value falls after last bin')
    integer_bins = np.searchsorted(index, datetime_bins, side=self.closed)
    counts = np.diff(integer_bins)
    codes = np.repeat(np.arange(len(labels)), counts)
    first_items = pd.Series(integer_bins[:-1], labels, copy=False)
    non_duplicate = ~first_items.duplicated('last')
    return (first_items.where(non_duplicate), codes)