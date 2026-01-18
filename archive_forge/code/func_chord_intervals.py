import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def chord_intervals(quality_str):
    """
    Convert a chord quality string to a pitch class representation. For
    example, 'maj' becomes [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0].

    Parameters
    ----------
    quality_str : str
        String defining the chord quality.

    Returns
    -------
    pitch_classes : numpy array
        Binary pitch class representation of chord quality.

    """
    list_idx = quality_str.find('(')
    if list_idx == -1:
        return _shorthands[quality_str].copy()
    if list_idx != 0:
        ivs = _shorthands[quality_str[:list_idx]].copy()
    else:
        ivs = np.zeros(12, dtype=np.int)
    return interval_list(quality_str[list_idx:], ivs)