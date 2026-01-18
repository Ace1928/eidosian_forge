from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import (evaluation_io, MultiClassEvaluation, SumEvaluation,
from .onsets import onset_evaluation, OnsetEvaluation
from ..io import load_notes
@property
def std_error(self):
    """Standard deviation of the errors."""
    warnings.warn('std_error is given for all notes, this will change!')
    return np.nanmean([e.std_error for e in self.eval_objects])