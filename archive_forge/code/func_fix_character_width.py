import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def fix_character_width(text):
    """
    The ASCII characters, katakana, and Hangul characters have alternate
    "halfwidth" or "fullwidth" forms that help text line up in a grid.

    If you don't need these width properties, you probably want to replace
    these characters with their standard form, which is what this function
    does.

    Note that this replaces the ideographic space, U+3000, with the ASCII
    space, U+20.

        >>> print(fix_character_width("ＬＯＵＤ\u3000ＮＯＩＳＥＳ"))
        LOUD NOISES
        >>> print(fix_character_width("Ｕﾀｰﾝ"))   # this means "U-turn"
        Uターン
    """
    return text.translate(WIDTH_MAP)