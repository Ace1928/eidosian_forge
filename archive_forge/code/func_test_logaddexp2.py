from itertools import product
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.printing.repr import srepr
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
def test_logaddexp2():
    lae2_xy = logaddexp2(x, y)
    ref2_xy = log(2 ** x + 2 ** y) / log(2)
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert (lae2_xy.diff(wrt, deriv_order) - ref2_xy.diff(wrt, deriv_order)).rewrite(log).cancel() == 0

    def lb(x):
        return log(x) / log(2)
    two_thirds = S.One * 2 / 3
    four_thirds = 2 * two_thirds
    lbTwoThirds = lb(two_thirds)
    lbFourThirds = lb(four_thirds)
    lae2_sum_to_2 = logaddexp2(lbTwoThirds, lbFourThirds)
    assert lae2_sum_to_2.rewrite(log) == 1
    assert lae2_sum_to_2.simplify() == 1
    was = logaddexp2(x, y)
    assert srepr(was) == srepr(was.simplify())