from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def get_conds(self, x, params, prev_conds=None):
    if prev_conds is None:
        prev_conds = [False] * len(self.condition_cb_pairs)
    return tuple([bw(x, params) if prev else fw(x, params) for prev, (fw, bw) in zip(prev_conds, self.condition_cb_pairs)])