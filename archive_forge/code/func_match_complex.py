from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def match_complex(target, x):
    target_iscomplex = anp.iscomplexobj(target)
    x_iscomplex = anp.iscomplexobj(x)
    if x_iscomplex and (not target_iscomplex):
        return anp.real(x)
    elif not x_iscomplex and target_iscomplex:
        return x + 0j
    else:
        return x