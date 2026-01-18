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
def create_state_multi_precision(self, index, weight):
    weight_master_copy = None
    if self.multi_precision and weight.dtype == numpy.float16:
        weight_master_copy = weight.astype(numpy.float32)
        return (self.create_state(index, weight_master_copy), weight_master_copy)
    if weight.dtype == numpy.float16 and (not self.multi_precision):
        warnings.warn('Accumulating with float16 in optimizer can lead to poor accuracy or slow convergence. Consider using multi_precision=True option of the NAG optimizer')
    return self.create_state(index, weight)