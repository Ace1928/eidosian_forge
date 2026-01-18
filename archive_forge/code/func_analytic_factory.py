from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def analytic_factory(x0, y0, p0, be):
    return {ori_sys.dep[tgt]: y0[ori_sys.dep[tgt] if ori_sys.dep_by_name else tgt] - sum([A[ri, ci] * (ori_sys.dep[ci] - y0[ori_sys.dep[ci] if ori_sys.dep_by_name else ci]) for ci in range(A.cols) if ci != tgt]) / A[ri, tgt] for ri, tgt in row_tgt}