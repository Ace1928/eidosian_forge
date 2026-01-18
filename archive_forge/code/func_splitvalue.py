from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def splitvalue(attr):
    warnings.warn('urllib.parse.splitvalue() is deprecated as of 3.8, use urllib.parse.parse_qsl() instead', DeprecationWarning, stacklevel=2)
    return _splitvalue(attr)