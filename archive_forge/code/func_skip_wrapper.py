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
@functools.wraps(test_item)
def skip_wrapper(*args, **kwargs):
    raise SkipTest(reason)