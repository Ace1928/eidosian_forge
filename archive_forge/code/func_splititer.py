import regex._regex_core as _regex_core
import regex._regex as _regex
from threading import RLock as _RLock
from locale import getpreferredencoding as _getpreferredencoding
from regex._regex_core import *
from regex._regex_core import (_ALL_VERSIONS, _ALL_ENCODINGS, _FirstSetError,
from regex._regex_core import (ALNUM as _ALNUM, Info as _Info, OP as _OP, Source
import copyreg as _copy_reg
def splititer(pattern, string, maxsplit=0, flags=0, concurrent=None, timeout=None, ignore_unused=False, **kwargs):
    """Return an iterator yielding the parts of a split string."""
    pat = _compile(pattern, flags, ignore_unused, kwargs, True)
    return pat.splititer(string, maxsplit, concurrent, timeout)