from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_chained_solvers():
    powell_numpy = NeqSys(2, 2, _powell)
    powell_mpmath = NeqSys(2, 2, _powell)
    _test_powell([(powell_numpy, None), (powell_mpmath, 'mpmath')])