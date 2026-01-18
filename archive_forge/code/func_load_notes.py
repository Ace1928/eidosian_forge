from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
@suppress_warnings
def load_notes(filename):
    """
    Load the notes from the given file, one note per line of format
    'onset_time' 'note_number' ['duration' ['velocity']].

    Parameters
    ----------
    filename: str or file handle
        File to load the notes from.

    Returns
    -------
    numpy array
        Notes.

    """
    return np.loadtxt(filename, ndmin=2)