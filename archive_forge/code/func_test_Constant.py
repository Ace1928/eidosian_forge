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
def test_Constant():
    c23 = Constant(2.3)
    assert abs(2.3 - float(c23)) < 1e-15
    c_div = Constant(2.3) / Constant(3.4)
    assert abs(2.3 / 3.4 - float(c_div)) < 1e-15