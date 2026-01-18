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
def stdin_get_value():
    """Read the value from stdin."""
    return TextIOWrapper(sys.stdin.buffer, errors='ignore').read()