from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def partially_solved_pre_processor(x, y, p):
    if y.ndim == 2:
        return zip(*[partially_solved_pre_processor(_x, _y, _p) for _x, _y, _p in zip(x, y, p)])
    return (x, _skip(self.ori_analyt_idx_map, y), _append(p, [x[0]], y))