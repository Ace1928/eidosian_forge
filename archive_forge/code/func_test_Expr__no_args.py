import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
def test_Expr__no_args():
    K1 = MyK(unique_keys=('H1', 'S1'))
    K2 = MyK(unique_keys=('H2', 'S2'))
    add = K1 + K2
    T = 298.15
    res = add({'H1': 2, 'H2': 3, 'S1': 5, 'S2': 7, 'T': T})
    RT = 8.3145 * 298.15
    ref = math.exp(-(2 - T * 5) / RT) + math.exp(-(3 - T * 7) / RT)
    assert abs(res - ref) < 1e-14