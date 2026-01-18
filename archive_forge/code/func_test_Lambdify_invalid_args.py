from __future__ import (absolute_import, division, print_function)
from functools import reduce
from operator import add, mul
import math
import numpy as np
import pytest
from pytest import raises
from .. import Backend
@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym',), backends))
def test_Lambdify_invalid_args(key):
    se = Backend(key)
    x = se.Symbol('x')
    log = se.Lambdify([x], [se.log(x)])
    div = se.Lambdify([x], [1 / x])
    assert math.isnan(log([-1])[0])
    assert math.isinf(-log([0])[0])
    assert math.isinf(div([0])[0])
    assert math.isinf(-div([-0])[0])