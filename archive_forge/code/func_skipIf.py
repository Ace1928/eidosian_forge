import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
def skipIf(condition, reason):
    """
    Skip a test if the condition is true.
    """
    if condition:
        return skip(reason)
    return _id