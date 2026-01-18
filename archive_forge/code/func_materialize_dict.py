import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def materialize_dict(x):
    return {k: maybe_materialize(v) for k, v in x.items()}