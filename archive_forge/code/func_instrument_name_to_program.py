import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def instrument_name_to_program(instrument_name):
    """Converts an instrument name to the corresponding General MIDI program
    number.  Conversion is case, whitespace, and non-alphanumeric character
    insensitive.

    Parameters
    ----------
    instrument_name : str
        Name of an instrument which exists in the general MIDI standard.
        If the instrument is not found, a ValueError is raised.

    Returns
    -------
    program_number : int
        The MIDI program number corresponding to this instrument.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """
    normalized_inst_name = __normalize_str(instrument_name)
    normalized_inst_names = [__normalize_str(name) for name in INSTRUMENT_MAP]
    try:
        program_number = normalized_inst_names.index(normalized_inst_name)
    except:
        raise ValueError('{} is not a valid General MIDI instrument name.'.format(instrument_name))
    return program_number