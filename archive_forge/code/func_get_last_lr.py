import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def get_last_lr(self):
    """ Return last computed learning rate by current scheduler.
        """
    return self._last_lr