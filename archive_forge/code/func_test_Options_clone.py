from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Options_clone():
    opt = Options((x, y, z), {'domain': 'ZZ'})
    assert opt.gens == (x, y, z)
    assert opt.domain == ZZ
    assert ('order' in opt) is False
    new_opt = opt.clone({'gens': (x, y), 'order': 'lex'})
    assert opt.gens == (x, y, z)
    assert opt.domain == ZZ
    assert ('order' in opt) is False
    assert new_opt.gens == (x, y)
    assert new_opt.domain == ZZ
    assert ('order' in new_opt) is True