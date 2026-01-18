from sympy.abc import t, w, x, y, z, n, k, m, p, i
from sympy.assumptions import (ask, AssumptionsContext, Q, register_handler,
from sympy.assumptions.assume import assuming, global_assumptions, Predicate
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.facts import (single_fact_lookup,
from sympy.assumptions.handlers import AskHandler
from sympy.assumptions.ask_generated import (get_all_known_facts,
from sympy.core.add import Add
from sympy.core.numbers import (I, Integer, Rational, oo, zoo, pi)
from sympy.core.singleton import S
from sympy.core.power import Pow
from sympy.core.symbol import Str, symbols, Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
from sympy.logic.boolalg import Equivalent, Implies, Xor, And, to_cnf
from sympy.matrices import Matrix, SparseMatrix
from sympy.testing.pytest import (XFAIL, slow, raises, warns_deprecated_sympy,
import math
def test_issue_7246():
    assert ask(Q.positive(atan(p)), Q.positive(p)) is True
    assert ask(Q.positive(atan(p)), Q.negative(p)) is False
    assert ask(Q.positive(atan(p)), Q.zero(p)) is False
    assert ask(Q.positive(atan(x))) is None
    assert ask(Q.positive(asin(p)), Q.positive(p)) is None
    assert ask(Q.positive(asin(p)), Q.zero(p)) is None
    assert ask(Q.positive(asin(Rational(1, 7)))) is True
    assert ask(Q.positive(asin(x)), Q.positive(x) & Q.nonpositive(x - 1)) is True
    assert ask(Q.positive(asin(x)), Q.negative(x) & Q.nonnegative(x + 1)) is False
    assert ask(Q.positive(acos(p)), Q.positive(p)) is None
    assert ask(Q.positive(acos(Rational(1, 7)))) is True
    assert ask(Q.positive(acos(x)), Q.nonnegative(x + 1) & Q.nonpositive(x - 1)) is True
    assert ask(Q.positive(acos(x)), Q.nonnegative(x - 1)) is None
    assert ask(Q.positive(acot(x)), Q.positive(x)) is True
    assert ask(Q.positive(acot(x)), Q.real(x)) is True
    assert ask(Q.positive(acot(x)), Q.imaginary(x)) is False
    assert ask(Q.positive(acot(x))) is None