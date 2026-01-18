import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def note_number_to_hz(note_number):
    """Convert a (fractional) MIDI note number to its frequency in Hz.

    Parameters
    ----------
    note_number : float
        MIDI note number, can be fractional.

    Returns
    -------
    note_frequency : float
        Frequency of the note in Hz.

    """
    return 440.0 * 2.0 ** ((note_number - 69) / 12.0)