from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
@wraps(metric)
def float_array(detections, annotations, *args, **kwargs):
    """Warp detections and annotations as numpy arrays."""
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    return metric(detections, annotations, *args, **kwargs)