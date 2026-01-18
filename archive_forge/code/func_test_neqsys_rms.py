from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_neqsys_rms():
    ns = NeqSys(2, 2, f)
    x = [[1, 0], [2, 1], [3, 2], [7, 4], [5, 13]]
    p = [3]
    rms = ns.rms(x, p)
    ref = [np.sqrt(np.sum(np.square(f(x[i], p))) / 2) for i in range(5)]
    assert np.allclose(rms, ref)