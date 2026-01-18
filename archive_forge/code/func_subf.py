import regex._regex_core as _regex_core
import regex._regex as _regex
from threading import RLock as _RLock
from locale import getpreferredencoding as _getpreferredencoding
from regex._regex_core import *
from regex._regex_core import (_ALL_VERSIONS, _ALL_ENCODINGS, _FirstSetError,
from regex._regex_core import (ALNUM as _ALNUM, Info as _Info, OP as _OP, Source
import copyreg as _copy_reg
def subf(pattern, format, string, count=0, flags=0, pos=None, endpos=None, concurrent=None, timeout=None, ignore_unused=False, **kwargs):
    """Return the string obtained by replacing the leftmost (or rightmost with a
    reverse pattern) non-overlapping occurrences of the pattern in string by the
    replacement format. format can be either a string or a callable; if a string,
    it's treated as a format string; if a callable, it's passed the match object
    and must return a replacement string to be used."""
    pat = _compile(pattern, flags, ignore_unused, kwargs, True)
    return pat.subf(format, string, count, pos, endpos, concurrent, timeout)