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
def python_3000_not_equal(logical_line):
    """New code should always use != instead of <>.

    The older syntax is removed in Python 3.

    Okay: if a != 'no':
    W603: if a <> 'no':
    """
    pos = logical_line.find('<>')
    if pos > -1:
        yield (pos, "W603 '<>' is deprecated, use '!='")