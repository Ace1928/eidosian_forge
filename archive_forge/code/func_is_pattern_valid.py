import re
from . import lazy_regex
from .trace import mutter, warning
@staticmethod
def is_pattern_valid(pattern):
    """Returns True if pattern is valid.

        :param pattern: Normalized pattern.
        is_pattern_valid() assumes pattern to be normalized.
        see: globbing.normalize_pattern
        """
    result = True
    translator = Globster.pattern_info[Globster.identify(pattern)]['translator']
    tpattern = '(%s)' % translator(pattern)
    try:
        re_obj = lazy_regex.lazy_compile(tpattern, re.UNICODE)
        re_obj.search('')
    except lazy_regex.InvalidPattern:
        result = False
    return result