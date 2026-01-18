import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def register_path_fn(name, fn):
    """Add path finding function ``fn`` as an option with ``name``.
    """
    if name in _PATH_OPTIONS:
        raise KeyError("Path optimizer '{}' already exists.".format(name))
    _PATH_OPTIONS[name.lower()] = fn