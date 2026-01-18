import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def get_exc_info():
    try:
        A, B, C = (Exception('A'), Exception('B'), Exception('C'))
        edges = [(C, B), (B, A), (A, C)]
        for ex1, ex2 in edges:
            ex1.__cause__ = ex2
            ex2.__context__ = ex1
        raise C
    except:
        return sys.exc_info()