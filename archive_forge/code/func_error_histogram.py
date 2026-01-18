from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
@property
def error_histogram(self):
    """Error histogram."""
    if not self.eval_objects:
        return np.zeros(0)
    return np.sum([e.error_histogram for e in self.eval_objects], axis=0)