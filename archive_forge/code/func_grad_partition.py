from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_partition(ans, x, kth, axis=-1, kind='introselect', order=None):
    if len(x.shape) > 1:
        raise NotImplementedError('Gradient of partition not implemented for multi-dimensional arrays.')
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return lambda g: unpermuter(g, partition_perm)