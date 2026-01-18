import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def is_better(self, a, best):
    if self.mode == 'min' and self.threshold_mode == 'rel':
        rel_epsilon = 1.0 - self.threshold
        return a < best * rel_epsilon
    elif self.mode == 'min' and self.threshold_mode == 'abs':
        return a < best - self.threshold
    elif self.mode == 'max' and self.threshold_mode == 'rel':
        rel_epsilon = self.threshold + 1.0
        return a > best * rel_epsilon
    else:
        return a > best + self.threshold