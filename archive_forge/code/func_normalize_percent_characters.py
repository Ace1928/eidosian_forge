import re
from . import compat
from . import misc
def normalize_percent_characters(s):
    """All percent characters should be upper-cased.

    For example, ``"%3afoo%DF%ab"`` should be turned into ``"%3Afoo%DF%AB"``.
    """
    matches = set(PERCENT_MATCHER.findall(s))
    for m in matches:
        if not m.isupper():
            s = s.replace(m, m.upper())
    return s