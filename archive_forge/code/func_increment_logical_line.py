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
def increment_logical_line(self):
    """Signal a new logical line."""
    self.counters['logical lines'] += 1