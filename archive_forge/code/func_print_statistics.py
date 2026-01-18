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
def print_statistics(self, prefix=''):
    """Print overall statistics (number of errors and warnings)."""
    for line in self.get_statistics(prefix):
        print(line)