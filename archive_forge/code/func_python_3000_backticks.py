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
def python_3000_backticks(logical_line):
    """Use repr() instead of backticks in Python 3.

    Okay: val = repr(1 + 2)
    W604: val = `1 + 2`
    """
    pos = logical_line.find('`')
    if pos > -1:
        yield (pos, "W604 backticks are deprecated, use 'repr()'")