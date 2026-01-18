from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Extension_preprocess():
    assert Extension.preprocess(True) is True
    assert Extension.preprocess(1) is True
    assert Extension.preprocess([]) is None
    assert Extension.preprocess(sqrt(2)) == {sqrt(2)}
    assert Extension.preprocess([sqrt(2)]) == {sqrt(2)}
    assert Extension.preprocess([sqrt(2), I]) == {sqrt(2), I}
    raises(OptionError, lambda: Extension.preprocess(False))
    raises(OptionError, lambda: Extension.preprocess(0))