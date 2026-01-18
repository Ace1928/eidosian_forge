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
def python_3000_raise_comma(logical_line):
    """When raising an exception, use "raise ValueError('message')".

    The older form is removed in Python 3.

    Okay: raise DummyError("Message")
    W602: raise DummyError, "Message"
    """
    match = RAISE_COMMA_REGEX.match(logical_line)
    if match and (not RERAISE_COMMA_REGEX.match(logical_line)):
        yield (match.end() - 1, 'W602 deprecated form of raising exception')