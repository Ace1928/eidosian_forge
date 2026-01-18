from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Domain_preprocess():
    assert Domain.preprocess(ZZ) == ZZ
    assert Domain.preprocess(QQ) == QQ
    assert Domain.preprocess(EX) == EX
    assert Domain.preprocess(FF(2)) == FF(2)
    assert Domain.preprocess(ZZ[x, y]) == ZZ[x, y]
    assert Domain.preprocess('Z') == ZZ
    assert Domain.preprocess('Q') == QQ
    assert Domain.preprocess('ZZ') == ZZ
    assert Domain.preprocess('QQ') == QQ
    assert Domain.preprocess('EX') == EX
    assert Domain.preprocess('FF(23)') == FF(23)
    assert Domain.preprocess('GF(23)') == GF(23)
    raises(OptionError, lambda: Domain.preprocess('Z[]'))
    assert Domain.preprocess('Z[x]') == ZZ[x]
    assert Domain.preprocess('Q[x]') == QQ[x]
    assert Domain.preprocess('R[x]') == RR[x]
    assert Domain.preprocess('C[x]') == CC[x]
    assert Domain.preprocess('ZZ[x]') == ZZ[x]
    assert Domain.preprocess('QQ[x]') == QQ[x]
    assert Domain.preprocess('RR[x]') == RR[x]
    assert Domain.preprocess('CC[x]') == CC[x]
    assert Domain.preprocess('Z[x,y]') == ZZ[x, y]
    assert Domain.preprocess('Q[x,y]') == QQ[x, y]
    assert Domain.preprocess('R[x,y]') == RR[x, y]
    assert Domain.preprocess('C[x,y]') == CC[x, y]
    assert Domain.preprocess('ZZ[x,y]') == ZZ[x, y]
    assert Domain.preprocess('QQ[x,y]') == QQ[x, y]
    assert Domain.preprocess('RR[x,y]') == RR[x, y]
    assert Domain.preprocess('CC[x,y]') == CC[x, y]
    raises(OptionError, lambda: Domain.preprocess('Z()'))
    assert Domain.preprocess('Z(x)') == ZZ.frac_field(x)
    assert Domain.preprocess('Q(x)') == QQ.frac_field(x)
    assert Domain.preprocess('ZZ(x)') == ZZ.frac_field(x)
    assert Domain.preprocess('QQ(x)') == QQ.frac_field(x)
    assert Domain.preprocess('Z(x,y)') == ZZ.frac_field(x, y)
    assert Domain.preprocess('Q(x,y)') == QQ.frac_field(x, y)
    assert Domain.preprocess('ZZ(x,y)') == ZZ.frac_field(x, y)
    assert Domain.preprocess('QQ(x,y)') == QQ.frac_field(x, y)
    assert Domain.preprocess('Q<I>') == QQ.algebraic_field(I)
    assert Domain.preprocess('QQ<I>') == QQ.algebraic_field(I)
    assert Domain.preprocess('Q<sqrt(2), I>') == QQ.algebraic_field(sqrt(2), I)
    assert Domain.preprocess('QQ<sqrt(2), I>') == QQ.algebraic_field(sqrt(2), I)
    raises(OptionError, lambda: Domain.preprocess('abc'))