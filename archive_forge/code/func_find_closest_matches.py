from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
def find_closest_matches(detections, annotations):
    """
    Find the closest annotation for each detection.

    Parameters
    ----------
    detections : list or numpy array
        Detected events.
    annotations : list or numpy array
        Annotated events.

    Returns
    -------
    indices : numpy array
        Indices of the closest matches.

    Notes
    -----
    The sequences must be ordered.

    """
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')
    if len(detections) == 0 or len(annotations) == 0:
        return np.zeros(0, dtype=np.int)
    if len(annotations) == 1:
        return np.zeros(len(detections), dtype=np.int)
    indices = annotations.searchsorted(detections)
    indices = np.clip(indices, 1, len(annotations) - 1)
    left = annotations[indices - 1]
    right = annotations[indices]
    indices -= detections - left < right - detections
    return indices