import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def join_array_dispatcher(*args, **kwargs):
    """Dispatcher for functions where first argument is a sequence."""
    try:
        return infer_backend(args[0][0])
    except (TypeError, ValueError):
        return infer_backend(args[0])