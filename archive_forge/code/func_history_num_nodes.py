import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def history_num_nodes(self):
    """Return the number of unique computational nodes in the history of
        this ``LazyArray``.
        """
    num_nodes = 0
    for _ in self.descend():
        num_nodes += 1
    return num_nodes