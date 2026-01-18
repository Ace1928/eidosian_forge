import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def key_number_to_key_name(key_number):
    """Convert a key number to a key string.

    Parameters
    ----------
    key_number : int
        Uses pitch classes to represent major and minor keys.
        For minor keys, adds a 12 offset.
        For example, C major is 0 and C minor is 12.

    Returns
    -------
    key_name : str
        Key name in the format ``'(root) (mode)'``, e.g. ``'Gb minor'``.
        Gives preference for keys with flats, with the exception of F#, G# and
        C# minor.
    """
    if not isinstance(key_number, int):
        raise ValueError('`key_number` is not int!')
    if not (key_number >= 0 and key_number < 24):
        raise ValueError('`key_number` is larger than 24')
    keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    key_idx = key_number % 12
    mode = key_number // 12
    if mode == 0:
        return keys[key_idx] + ' Major'
    elif mode == 1:
        if key_idx in [1, 6, 8]:
            return keys[key_idx - 1] + '# minor'
        else:
            return keys[key_idx] + ' minor'