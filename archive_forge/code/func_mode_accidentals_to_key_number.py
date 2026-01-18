import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def mode_accidentals_to_key_number(mode, num_accidentals):
    """Convert a given number of accidentals and mode to a key number.

    Parameters
    ----------
    mode : int
        0 is major, 1 is minor.
    num_accidentals : int
        Positive number is used for sharps, negative number is used for flats.

    Returns
    -------
    key_number : int
        Integer representing the key and its mode.
    """
    if not (isinstance(num_accidentals, int) and num_accidentals > -8 and (num_accidentals < 8)):
        raise ValueError('Number of accidentals {} is not valid'.format(num_accidentals))
    if mode not in (0, 1):
        raise ValueError('Mode {} is not recognizable, must be 0 or 1'.format(mode))
    sharp_keys = 'CGDAEBF'
    flat_keys = 'FBEADGC'
    if num_accidentals >= 0:
        num_sharps = num_accidentals // 6
        key = sharp_keys[num_accidentals % 7] + '#' * int(num_sharps)
    elif num_accidentals == -1:
        key = 'F'
    else:
        key = flat_keys[(-1 * num_accidentals - 1) % 7] + 'b'
    key += ' Major'
    key_number = key_name_to_key_number(key)
    if mode == 1:
        key_number = 12 + (key_number - 3) % 12
    return key_number