from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
def find_longest_continuous_segment(sequence_indices):
    """
    ind the longest consecutive segment in the given sequence.

    Parameters
    ----------
    sequence_indices : numpy array
        Indices of the beats

    Returns
    -------
    length : int
        Length of the longest consecutive segment.
    start : int
        Start position of the longest continuous segment.

    """
    boundaries = np.nonzero(np.diff(sequence_indices) != 1)[0] + 1
    boundaries = np.concatenate(([0], boundaries, [len(sequence_indices)]))
    segment_lengths = np.diff(boundaries)
    length = int(np.max(segment_lengths))
    start_pos = int(boundaries[np.argmax(segment_lengths)])
    return (length, start_pos)