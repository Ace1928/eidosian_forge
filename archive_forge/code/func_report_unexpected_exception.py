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
def report_unexpected_exception(self, out, test, example, exc_info):
    raise UnexpectedException(test, example, exc_info)