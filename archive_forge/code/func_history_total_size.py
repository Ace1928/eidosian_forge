import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def history_total_size(self):
    """The the total size of all unique arrays in the computational graph,
        possibly relevant e.g. for back-propagation algorithms.
        """
    return sum((node.size for node in self.descend()))