from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
def rq_tuple(tpl, name):
    """ Check if chunks/maxshape match dataset rank """
    if tpl in (None, True):
        return
    try:
        tpl = tuple(tpl)
    except TypeError:
        raise TypeError('"%s" argument must be None or a sequence object' % name)
    if len(tpl) != len(shape):
        raise ValueError('"%s" must have same rank as dataset shape' % name)