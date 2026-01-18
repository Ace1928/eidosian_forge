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
def test_complex_infinity():
    assert ask(Q.commutative(zoo)) is True
    assert ask(Q.integer(zoo)) is False
    assert ask(Q.rational(zoo)) is False
    assert ask(Q.algebraic(zoo)) is False
    assert ask(Q.real(zoo)) is False
    assert ask(Q.extended_real(zoo)) is False
    assert ask(Q.complex(zoo)) is False
    assert ask(Q.irrational(zoo)) is False
    assert ask(Q.imaginary(zoo)) is False
    assert ask(Q.positive(zoo)) is False
    assert ask(Q.negative(zoo)) is False
    assert ask(Q.zero(zoo)) is False
    assert ask(Q.nonzero(zoo)) is False
    assert ask(Q.even(zoo)) is False
    assert ask(Q.odd(zoo)) is False
    assert ask(Q.finite(zoo)) is False
    assert ask(Q.infinite(zoo)) is True
    assert ask(Q.prime(zoo)) is False
    assert ask(Q.composite(zoo)) is False
    assert ask(Q.hermitian(zoo)) is False
    assert ask(Q.antihermitian(zoo)) is False
    assert ask(Q.positive_infinite(zoo)) is False
    assert ask(Q.negative_infinite(zoo)) is False