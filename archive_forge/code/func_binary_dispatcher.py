import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def binary_dispatcher(*args, **_):
    """There are cases when we want to take into account both backends of two
    arguments, e.g. a lazy variable and a constant array.
    """
    return infer_backend_multi(*args[:2])