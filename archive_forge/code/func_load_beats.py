from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
@suppress_warnings
def load_beats(filename, downbeats=False):
    """
    Load the beats from the given file, one beat per line of format
    'beat_time' ['beat_number'].

    Parameters
    ----------
    filename : str or file handle
        File to load the beats from.
    downbeats : bool, optional
        Load only downbeats instead of beats.

    Returns
    -------
    numpy array
        Beats.

    """
    values = np.loadtxt(filename, ndmin=1)
    if values.ndim > 1:
        if downbeats:
            return values[values[:, 1] == 1][:, 0]
        else:
            return values[:, 0]
    return values