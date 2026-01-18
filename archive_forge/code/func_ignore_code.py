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
def ignore_code(self, code):
    """Check if the error code should be ignored.

        If 'options.select' contains a prefix of the error code,
        return False.  Else, if 'options.ignore' contains a prefix of
        the error code, return True.
        """
    if len(code) < 4 and any((s.startswith(code) for s in self.options.select)):
        return False
    return code.startswith(self.options.ignore) and (not code.startswith(self.options.select))