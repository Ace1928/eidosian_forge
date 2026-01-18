from contextlib import contextmanager
import numpy as np
from_record_like = None
def check_array_compatibility(ary1, ary2):
    ary1sq, ary2sq = (ary1.squeeze(), ary2.squeeze())
    if ary1.dtype != ary2.dtype:
        raise TypeError('incompatible dtype: %s vs. %s' % (ary1.dtype, ary2.dtype))
    if ary1sq.shape != ary2sq.shape:
        raise ValueError('incompatible shape: %s vs. %s' % (ary1.shape, ary2.shape))
    if ary1sq.strides != ary2sq.strides:
        raise ValueError('incompatible strides: %s vs. %s' % (ary1.strides, ary2.strides))