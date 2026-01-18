from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.compat._optional import import_optional_dependency
def run_ewm(self, weighted_avg, deltas, min_periods, ewm_func):
    result, old_wt = ewm_func(weighted_avg, deltas, min_periods, self.old_wt_factor, self.new_wt, self.old_wt, self.adjust, self.ignore_na)
    self.old_wt = old_wt
    self.last_ewm = result[-1]
    return result