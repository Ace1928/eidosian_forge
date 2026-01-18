from sympy.assumptions.ask import Q
from sympy.assumptions.assume import assuming
from sympy.core.numbers import (I, pi)
from sympy.core.relational import (Eq, Gt)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import Implies
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.satask import (satask, extract_predargs,
from sympy.testing.pytest import raises, XFAIL
def test_extract_predargs():
    props = CNF.from_prop(Q.zero(Abs(x * y)) & Q.zero(x * y))
    assump = CNF.from_prop(Q.zero(x))
    context = CNF.from_prop(Q.zero(y))
    assert extract_predargs(props) == {Abs(x * y), x * y}
    assert extract_predargs(props, assump) == {Abs(x * y), x * y, x}
    assert extract_predargs(props, assump, context) == {Abs(x * y), x * y, x, y}
    props = CNF.from_prop(Eq(x, y))
    assump = CNF.from_prop(Gt(y, z))
    assert extract_predargs(props, assump) == {x, y, z}