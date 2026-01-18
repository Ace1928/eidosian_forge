from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
def calc_intervals(events, fwd=False):
    """
    Calculate the intervals of all events to the previous/next event.

    Parameters
    ----------
    events : numpy array
        Beat sequence.
    fwd : bool, optional
        Calculate the intervals towards the next event (instead of previous).

    Returns
    -------
    numpy array
        Beat intervals.

    Notes
    -----
    The sequence must be ordered. The first (last) interval will be set to
    the same value as the second (second to last) interval (when used in
    `fwd` mode).

    """
    if len(events) < 2:
        raise BeatIntervalError
    interval = np.zeros_like(events)
    if fwd:
        interval[:-1] = np.diff(events)
        interval[-1] = interval[-2]
    else:
        interval[1:] = np.diff(events)
        interval[0] = interval[1]
    return interval