import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def drum_name_to_note_number(drum_name):
    """Converts a drum name to the corresponding MIDI note number for a
    percussion instrument.  Conversion is case, whitespace, and
    non-alphanumeric character insensitive.

    Parameters
    ----------
    drum_name : str
        Name of a drum which exists in the general MIDI standard.
        If the drum is not found, a ValueError is raised.

    Returns
    -------
    note_number : int
        The MIDI note number corresponding to this drum.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """
    normalized_drum_name = __normalize_str(drum_name)
    normalized_drum_names = [__normalize_str(name) for name in DRUM_MAP]
    try:
        note_index = normalized_drum_names.index(normalized_drum_name)
    except:
        raise ValueError('{} is not a valid General MIDI drum name.'.format(drum_name))
    return note_index + 35