import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def fix_surrogates(text):
    """
    Replace 16-bit surrogate codepoints with the characters they represent
    (when properly paired), or with � otherwise.

        >>> high_surrogate = chr(0xd83d)
        >>> low_surrogate = chr(0xdca9)
        >>> print(fix_surrogates(high_surrogate + low_surrogate))
        💩
        >>> print(fix_surrogates(low_surrogate + high_surrogate))
        ��

    The above doctest had to be very carefully written, because even putting
    the Unicode escapes of the surrogates in the docstring was causing
    various tools to fail, which I think just goes to show why this fixer is
    necessary.
    """
    if SURROGATE_RE.search(text):
        text = SURROGATE_PAIR_RE.sub(convert_surrogate_pair, text)
        text = SURROGATE_RE.sub('�', text)
    return text