from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def guess_bytes(bstring: bytes) -> Tuple[str, str]:
    """
    NOTE: Using `guess_bytes` is not the recommended way of using ftfy. ftfy
    is not designed to be an encoding detector.

    In the unfortunate situation that you have some bytes in an unknown
    encoding, ftfy can guess a reasonable strategy for decoding them, by trying
    a few common encodings that can be distinguished from each other.

    Unlike the rest of ftfy, this may not be accurate, and it may *create*
    Unicode problems instead of solving them!

    The encodings we try here are:

    - UTF-16 with a byte order mark, because a UTF-16 byte order mark looks
      like nothing else
    - UTF-8, because it's the global standard, which has been used by a
      majority of the Web since 2008
    - "utf-8-variants", or buggy implementations of UTF-8
    - MacRoman, because Microsoft Office thinks it's still a thing, and it
      can be distinguished by its line breaks. (If there are no line breaks in
      the string, though, you're out of luck.)
    - "sloppy-windows-1252", the Latin-1-like encoding that is the most common
      single-byte encoding.
    """
    if isinstance(bstring, str):
        raise UnicodeError('This string was already decoded as Unicode. You should pass bytes to guess_bytes, not Unicode.')
    if bstring.startswith(b'\xfe\xff') or bstring.startswith(b'\xff\xfe'):
        return (bstring.decode('utf-16'), 'utf-16')
    byteset = set(bstring)
    try:
        if 237 in byteset or 192 in byteset:
            return (bstring.decode('utf-8-variants'), 'utf-8-variants')
        else:
            return (bstring.decode('utf-8'), 'utf-8')
    except UnicodeDecodeError:
        pass
    if 13 in byteset and 10 not in byteset:
        return (bstring.decode('macroman'), 'macroman')
    return (bstring.decode('sloppy-windows-1252'), 'sloppy-windows-1252')