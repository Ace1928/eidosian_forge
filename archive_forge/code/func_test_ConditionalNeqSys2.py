from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_ConditionalNeqSys2():
    _check_NaCl(_get_cneqsys2(), [(1, 1, 1), (1, 1, 0), (2, 2, 0), (1, 1, 3)])