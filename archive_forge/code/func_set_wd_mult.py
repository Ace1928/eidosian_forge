import logging
import math
import pickle
import warnings
import os
import numpy
from ..base import py_str
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
from ..ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
from ..ndarray.contrib import (multi_lamb_update, multi_mp_lamb_update)
from ..ndarray import sparse
from ..random import normal
from ..util import is_np_array
def set_wd_mult(self, args_wd_mult):
    self.wd_mult = {}
    for n in self.idx2name.values():
        is_weight = n.endswith('_weight')
        if not is_weight:
            self.wd_mult[n] = 0.0
    if self.sym_info:
        attr, arg_names = self.sym_info
        for name in arg_names:
            if name in attr and '__wd_mult__' in attr[name]:
                self.wd_mult[name] = float(attr[name]['__wd_mult__'])
    self.wd_mult.update(args_wd_mult)