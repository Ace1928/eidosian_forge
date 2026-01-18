from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
def get_freq(self):
    """Find the appropriate frequency string to describe the inferred frequency of self.index

        Adapted from `pandas.tsseries.frequencies._FrequencyInferer.get_freq` for CFTimeIndexes.

        Returns
        -------
        str or None
        """
    if not self.is_monotonic or not self.index.is_unique:
        return None
    delta = self.deltas[0]
    if _is_multiple(delta, _ONE_DAY):
        return self._infer_daily_rule()
    elif not len(self.deltas) == 1:
        return None
    if _is_multiple(delta, _ONE_HOUR):
        return _maybe_add_count('h', delta / _ONE_HOUR)
    elif _is_multiple(delta, _ONE_MINUTE):
        return _maybe_add_count('min', delta / _ONE_MINUTE)
    elif _is_multiple(delta, _ONE_SECOND):
        return _maybe_add_count('s', delta / _ONE_SECOND)
    elif _is_multiple(delta, _ONE_MILLI):
        return _maybe_add_count('ms', delta / _ONE_MILLI)
    else:
        return _maybe_add_count('us', delta / _ONE_MICRO)