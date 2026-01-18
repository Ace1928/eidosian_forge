import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def scale_fn(self, x):
    if self._scale_fn_custom is not None:
        return self._scale_fn_custom(x)
    else:
        return self._scale_fn_ref(x)