import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def fix_c1_controls(text):
    """
    If text still contains C1 control characters, treat them as their
    Windows-1252 equivalents. This matches what Web browsers do.
    """
    return C1_CONTROL_RE.sub(_c1_fixer, text)