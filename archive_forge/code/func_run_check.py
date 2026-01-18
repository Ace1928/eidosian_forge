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
def run_check(self, check, argument_names):
    """Run a check plugin."""
    arguments = []
    for name in argument_names:
        arguments.append(getattr(self, name))
    return check(*arguments)