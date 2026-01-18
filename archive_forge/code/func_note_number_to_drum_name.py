import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def note_number_to_drum_name(note_number):
    """Converts a MIDI note number in a percussion instrument to the
    corresponding drum name, according to the General MIDI standard.

    Any MIDI note number outside of the valid range (note 35-81, zero-indexed)
    will result in an empty string.

    Parameters
    ----------
    note_number : int
        MIDI note number.  If not an int, it will be rounded.

    Returns
    -------
    drum_name : str
        Name of the drum for this note for a percussion instrument.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """
    note_number = int(np.round(note_number))
    if note_number < 35 or note_number > 81:
        return ''
    else:
        return DRUM_MAP[note_number - 35]