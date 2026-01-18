import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def interval_list(intervals_str, given_pitch_classes=None):
    """
    Convert a list of intervals given as string to a binary pitch class
    representation. For example, 'b3, 5' would become
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0].

    Parameters
    ----------
    intervals_str : str
        List of intervals as comma-separated string (e.g. 'b3, 5').
    given_pitch_classes : None or numpy array
        If None, start with empty pitch class array, if numpy array of length
        12, this array will be modified.

    Returns
    -------
    pitch_classes : numpy array
        Binary pitch class representation of intervals.

    """
    if given_pitch_classes is None:
        given_pitch_classes = np.zeros(12, dtype=np.int)
    for int_def in intervals_str[1:-1].split(','):
        int_def = int_def.strip()
        if int_def[0] == '*':
            given_pitch_classes[interval(int_def[1:])] = 0
        else:
            given_pitch_classes[interval(int_def)] = 1
    return given_pitch_classes