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
def enterModuleContext(cm):
    """Same as enterContext, but module-wide."""
    return _enter_context(cm, addModuleCleanup)