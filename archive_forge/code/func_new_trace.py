import warnings
from contextlib import contextmanager
from collections import defaultdict
from .util import subvals, toposort
from .wrap_util import wraps
@contextmanager
def new_trace(self):
    self.top += 1
    yield self.top
    self.top -= 1