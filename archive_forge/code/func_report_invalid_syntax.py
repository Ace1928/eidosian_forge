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
def report_invalid_syntax(self):
    """Check if the syntax is valid."""
    exc_type, exc = sys.exc_info()[:2]
    if len(exc.args) > 1:
        offset = exc.args[1]
        if len(offset) > 2:
            offset = offset[1:3]
    else:
        offset = (1, 0)
    self.report_error(offset[0], offset[1] or 0, 'E901 %s: %s' % (exc_type.__name__, exc.args[0]), self.report_invalid_syntax)