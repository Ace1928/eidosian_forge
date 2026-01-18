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
@requires(parsing_library)
def test_Expr():
    cv = _get_cv()
    _ref = 0.8108020083055849
    assert abs(cv['Al']({'temperature': 273.15, 'molar_gas_constant': 8.3145}) - _ref) < 1e-14
    with pytest.raises(Exception):
        float(cv['Al'])