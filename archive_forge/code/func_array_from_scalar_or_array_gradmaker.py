from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def array_from_scalar_or_array_gradmaker(ans, array_args, array_kwargs, scarray):
    ndmin = array_kwargs.get('ndmin', 0)
    scarray_ndim = anp.ndim(scarray)
    if ndmin > scarray_ndim:
        return lambda g: anp.squeeze(g, axis=tuple(range(ndmin - scarray_ndim)))
    else:
        return lambda g: g