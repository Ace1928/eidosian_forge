import warnings
from contextlib import contextmanager
from collections import defaultdict
from .util import subvals, toposort
from .wrap_util import wraps
@classmethod
def new_root(cls, *args, **kwargs):
    root = cls.__new__(cls)
    root.initialize_root(*args, **kwargs)
    return root