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
def assertNoLogs(self, logger=None, level=None):
    """ Fail unless no log messages of level *level* or higher are emitted
        on *logger_name* or its children.

        This method must be used as a context manager.
        """
    from ._log import _AssertLogsContext
    return _AssertLogsContext(self, logger, level, no_logs=True)