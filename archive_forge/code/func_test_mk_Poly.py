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
def test_mk_Poly():
    Poly = mk_Poly('T', reciprocal=True)
    p = Poly([3, 2, 5, 7, 8, 2, 9])
    assert p.eval_poly({'T': 13}) == 2.57829
    assert p.parameter_keys == ('T',)