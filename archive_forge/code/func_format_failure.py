import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def format_failure(self, err):
    return 'Failed doctest test for %s\n  File "%s", line 0\n\n%s' % (self._dt_test.name, self._dt_test.filename, err)