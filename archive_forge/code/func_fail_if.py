import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
@contextlib.contextmanager
def fail_if(self):
    all_fails = compound()
    all_fails.fails.update(self.skips.union(self.fails))
    try:
        yield
    except Exception as ex:
        all_fails._expect_failure(config._current, ex)
    else:
        all_fails._expect_success(config._current)