from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def comparison_type(logical_line, noqa):
    """Object type comparisons should always use isinstance().

    Do not compare types directly.

    Okay: if isinstance(obj, int):
    E721: if type(obj) is type(1):

    When checking if an object is a string, keep in mind that it might be a
    unicode string too! In Python 2.3, str and unicode have a common base
    class, basestring, so you can do:

    Okay: if isinstance(obj, basestring):
    Okay: if type(a1) is type(b1):
    """
    match = COMPARE_TYPE_REGEX.search(logical_line)
    if match and (not noqa):
        inst = match.group(1)
        if inst and isidentifier(inst) and (inst not in SINGLETONS):
            return
        yield (match.start(), "E721 do not compare types, use 'isinstance()'")