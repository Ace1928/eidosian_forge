import pickle
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar, Replacer
def test_matchpy_connector():
    if matchpy is None:
        skip('matchpy not installed')
    from multiset import Multiset
    from matchpy import Pattern, Substitution
    w_ = WildDot('w_')
    w__ = WildPlus('w__')
    w___ = WildStar('w___')
    expr = x + y
    pattern = x + w_
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w_': y})
    expr = x + y + z
    pattern = x + w__
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w__': Multiset([y, z])})
    expr = x + y + z
    pattern = x + y + z + w___
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w___': Multiset()})