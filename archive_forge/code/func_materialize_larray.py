import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def materialize_larray(x):
    return x._materialize()