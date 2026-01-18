import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def decode_inconsistent_utf8(text):
    """
    Sometimes, text from one encoding ends up embedded within text from a
    different one. This is common enough that we need to be able to fix it.

    This is used as a transcoder within `fix_encoding`.
    """

    def fix_embedded_mojibake(match):
        substr = match.group(0)
        if len(substr) < len(text) and is_bad(substr):
            return ftfy.fix_encoding(substr)
        else:
            return substr
    return UTF8_DETECTOR_RE.sub(fix_embedded_mojibake, text)