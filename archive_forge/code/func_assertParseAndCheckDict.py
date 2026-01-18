import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
def assertParseAndCheckDict(self, expr, test_string, expected_dict, msg=None, verbose=True):
    """
            Convenience wrapper assert to test a parser element and input string, and assert that
            the resulting ParseResults.asDict() is equal to the expected_dict.
            """
    result = expr.parseString(test_string, parseAll=True)
    if verbose:
        print(result.dump())
    self.assertParseResultsEquals(result, expected_dict=expected_dict, msg=msg)