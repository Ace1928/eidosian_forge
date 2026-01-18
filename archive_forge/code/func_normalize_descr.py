import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def normalize_descr(descr):
    """Normalize a description adding the platform byteorder."""
    out = []
    for item in descr:
        dtype = item[1]
        if isinstance(dtype, str):
            if dtype[0] not in ['|', '<', '>']:
                onebyte = dtype[1:] == '1'
                if onebyte or dtype[0] in ['S', 'V', 'b']:
                    dtype = '|' + dtype
                else:
                    dtype = byteorder + dtype
            if len(item) > 2 and np.prod(item[2]) > 1:
                nitem = (item[0], dtype, item[2])
            else:
                nitem = (item[0], dtype)
            out.append(nitem)
        elif isinstance(dtype, list):
            l = normalize_descr(dtype)
            out.append((item[0], l))
        else:
            raise ValueError('Expected a str or list and got %s' % type(item))
    return out