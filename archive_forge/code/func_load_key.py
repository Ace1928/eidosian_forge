from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def load_key(filename):
    """
    Load the key from the given file.

    Parameters
    ----------
    filename : str or file handle
        File to read key information from.

    Returns
    -------
    str
        Key.

    """
    with open_file(filename) as f:
        return f.read().strip()