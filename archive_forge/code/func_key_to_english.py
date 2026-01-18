from __future__ import print_function
import binascii
from Cryptodome.Util.py3compat import bord, bchr
def key_to_english(key):
    """Transform an arbitrary key into a string containing English words.

    Example::

        >>> from Cryptodome.Util.RFC1751 import key_to_english
        >>> key_to_english(b'66666666')
        'RAM LOIS GOAD CREW CARE HIT'

    Args:
      key (byte string):
        The key to convert. Its length must be a multiple of 8.
    Return:
      A string of English words.
    """
    if len(key) % 8 != 0:
        raise ValueError('The length of the key must be a multiple of 8.')
    english = ''
    for index in range(0, len(key), 8):
        subkey = key[index:index + 8]
        skbin = _key2bin(subkey)
        p = 0
        for i in range(0, 64, 2):
            p = p + _extract(skbin, i, 2)
        skbin = _key2bin(subkey + bchr(p << 6 & 255))
        for i in range(0, 64, 11):
            english = english + wordlist[_extract(skbin, i, 11)] + ' '
    return english.strip()