from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
@property
def cmlt(self):
    """CMLt."""
    return np.nanmean([e.cmlt for e in self.eval_objects])