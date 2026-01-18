from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def splitnport(host, defport=-1):
    warnings.warn('urllib.parse.splitnport() is deprecated as of 3.8, use urllib.parse.urlparse() instead', DeprecationWarning, stacklevel=2)
    return _splitnport(host, defport)