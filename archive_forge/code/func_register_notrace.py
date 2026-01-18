import warnings
from contextlib import contextmanager
from collections import defaultdict
from .util import subvals, toposort
from .wrap_util import wraps
def register_notrace(trace_type, primitive_fun):
    notrace_primitives[trace_type].add(primitive_fun)