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
def assertParseResultsEquals(self, result, expected_list=None, expected_dict=None, msg=None):
    """
            Unit test assertion to compare a ParseResults object with an optional expected_list,
            and compare any defined results names with an optional expected_dict.
            """
    if expected_list is not None:
        self.assertEqual(expected_list, result.asList(), msg=msg)
    if expected_dict is not None:
        self.assertEqual(expected_dict, result.asDict(), msg=msg)