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
def report_start(self, out, test, example):
    """
        Report that the test runner is about to process the given
        example.  (Only displays a message if verbose=True)
        """
    if self._verbose:
        if example.want:
            out('Trying:\n' + _indent(example.source) + 'Expecting:\n' + _indent(example.want))
        else:
            out('Trying:\n' + _indent(example.source) + 'Expecting nothing\n')