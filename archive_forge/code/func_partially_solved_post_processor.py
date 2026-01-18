from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def partially_solved_post_processor(x, y, p):
    try:
        y[0][0, 0]
    except:
        pass
    else:
        return zip(*[partially_solved_post_processor(_x, _y, _p) for _x, _y, _p in zip(x, y, p)])
    new_y = np.empty(y.shape[:-1] + (y.shape[-1] + len(self.analytic_exprs),))
    analyt_y = self.analytic_cb(x, y, p)
    for idx in range(self._ori_sys.ny):
        if idx in self.ori_analyt_idx_map:
            new_y[..., idx] = analyt_y[..., self.ori_analyt_idx_map[idx]]
        else:
            new_y[..., idx] = y[..., self.ori_remaining_idx_map[idx]]
    return (x, new_y, p[:-(1 + self._ori_sys.ny)])