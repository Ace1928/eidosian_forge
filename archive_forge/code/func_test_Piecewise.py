import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_Piecewise():
    expr = sy.Piecewise((0, x < 0), (x, x < 2), (1, True))
    result = theano_code_(expr)
    assert result.owner.op == tt.switch
    expected = tt.switch(xt < 0, 0, tt.switch(xt < 2, xt, 1))
    assert theq(result, expected)
    expr = sy.Piecewise((x, x < 0))
    result = theano_code_(expr)
    expected = tt.switch(xt < 0, xt, np.nan)
    assert theq(result, expected)
    expr = sy.Piecewise((0, sy.And(x > 0, x < 2)), (x, sy.Or(x > 2, x < 0)))
    result = theano_code_(expr)
    expected = tt.switch(tt.and_(xt > 0, xt < 2), 0, tt.switch(tt.or_(xt > 2, xt < 0), xt, np.nan))
    assert theq(result, expected)