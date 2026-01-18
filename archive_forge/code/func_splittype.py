from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def splittype(url):
    warnings.warn('urllib.parse.splittype() is deprecated as of 3.8, use urllib.parse.urlparse() instead', DeprecationWarning, stacklevel=2)
    return _splittype(url)