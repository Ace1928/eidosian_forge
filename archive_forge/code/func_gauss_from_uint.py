import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def gauss_from_uint(x, n, bits):
    if bits in (64, 63):
        doubles = uniform_from_uint64(x)
    elif bits == 32:
        doubles = uniform_from_uint32(x)
    else:
        doubles = uniform_from_dsfmt(x)
    gauss = []
    loc = 0
    x1 = x2 = 0.0
    while len(gauss) < n:
        r2 = 2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * doubles[loc] - 1.0
            x2 = 2.0 * doubles[loc + 1] - 1.0
            r2 = x1 * x1 + x2 * x2
            loc += 2
        f = np.sqrt(-2.0 * np.log(r2) / r2)
        gauss.append(f * x2)
        gauss.append(f * x1)
    return gauss[:n]