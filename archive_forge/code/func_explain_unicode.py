from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def explain_unicode(text: str):
    """
    A utility method that's useful for debugging mysterious Unicode.

    It breaks down a string, showing you for each codepoint its number in
    hexadecimal, its glyph, its category in the Unicode standard, and its name
    in the Unicode standard.

        >>> explain_unicode('(╯°□°)╯︵ ┻━┻')
        U+0028  (       [Ps] LEFT PARENTHESIS
        U+256F  ╯       [So] BOX DRAWINGS LIGHT ARC UP AND LEFT
        U+00B0  °       [So] DEGREE SIGN
        U+25A1  □       [So] WHITE SQUARE
        U+00B0  °       [So] DEGREE SIGN
        U+0029  )       [Pe] RIGHT PARENTHESIS
        U+256F  ╯       [So] BOX DRAWINGS LIGHT ARC UP AND LEFT
        U+FE35  ︵      [Ps] PRESENTATION FORM FOR VERTICAL LEFT PARENTHESIS
        U+0020          [Zs] SPACE
        U+253B  ┻       [So] BOX DRAWINGS HEAVY UP AND HORIZONTAL
        U+2501  ━       [So] BOX DRAWINGS HEAVY HORIZONTAL
        U+253B  ┻       [So] BOX DRAWINGS HEAVY UP AND HORIZONTAL
    """
    for char in text:
        if char.isprintable():
            display = char
        else:
            display = char.encode('unicode-escape').decode('ascii')
        print('U+{code:04X}  {display} [{category}] {name}'.format(display=display_ljust(display, 7), code=ord(char), category=unicodedata.category(char), name=unicodedata.name(char, '<unknown>')))