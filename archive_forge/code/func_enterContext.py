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
def enterContext(self, cm):
    """Enters the supplied context manager.

        If successful, also adds its __exit__ method as a cleanup
        function and returns the result of the __enter__ method.
        """
    return _enter_context(cm, self.addCleanup)