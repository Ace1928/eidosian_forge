from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
def calc_absolute_errors(detections, annotations, matches=None):
    """
    Absolute errors of the detections to the closest annotations.

    Parameters
    ----------
    detections : list or numpy array
        Detected events.
    annotations : list or numpy array
        Annotated events.
    matches : list or numpy array
        Indices of the closest events.

    Returns
    -------
    errors : numpy array
        Absolute errors.

    Notes
    -----
    The sequences must be ordered. To speed up the calculation, a list of
    pre-computed indices of the closest matches can be used.

    """
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if matches is not None:
        matches = np.asarray(matches, dtype=np.int)
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')
    return np.abs(calc_errors(detections, annotations, matches))