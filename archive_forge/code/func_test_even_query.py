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
def test_even_query():
    assert ask(Q.even(x)) is None
    assert ask(Q.even(x), Q.integer(x)) is None
    assert ask(Q.even(x), ~Q.integer(x)) is False
    assert ask(Q.even(x), Q.rational(x)) is None
    assert ask(Q.even(x), Q.positive(x)) is None
    assert ask(Q.even(2 * x)) is None
    assert ask(Q.even(2 * x), Q.integer(x)) is True
    assert ask(Q.even(2 * x), Q.even(x)) is True
    assert ask(Q.even(2 * x), Q.irrational(x)) is False
    assert ask(Q.even(2 * x), Q.odd(x)) is True
    assert ask(Q.even(2 * x), ~Q.integer(x)) is None
    assert ask(Q.even(3 * x), Q.integer(x)) is None
    assert ask(Q.even(3 * x), Q.even(x)) is True
    assert ask(Q.even(3 * x), Q.odd(x)) is False
    assert ask(Q.even(x + 1), Q.odd(x)) is True
    assert ask(Q.even(x + 1), Q.even(x)) is False
    assert ask(Q.even(x + 2), Q.odd(x)) is False
    assert ask(Q.even(x + 2), Q.even(x)) is True
    assert ask(Q.even(7 - x), Q.odd(x)) is True
    assert ask(Q.even(7 + x), Q.odd(x)) is True
    assert ask(Q.even(x + y), Q.odd(x) & Q.odd(y)) is True
    assert ask(Q.even(x + y), Q.odd(x) & Q.even(y)) is False
    assert ask(Q.even(x + y), Q.even(x) & Q.even(y)) is True
    assert ask(Q.even(2 * x + 1), Q.integer(x)) is False
    assert ask(Q.even(2 * x * y), Q.rational(x) & Q.rational(x)) is None
    assert ask(Q.even(2 * x * y), Q.irrational(x) & Q.irrational(x)) is None
    assert ask(Q.even(x + y + z), Q.odd(x) & Q.odd(y) & Q.even(z)) is True
    assert ask(Q.even(x + y + z + t), Q.odd(x) & Q.odd(y) & Q.even(z) & Q.integer(t)) is None
    assert ask(Q.even(Abs(x)), Q.even(x)) is True
    assert ask(Q.even(Abs(x)), ~Q.even(x)) is None
    assert ask(Q.even(re(x)), Q.even(x)) is True
    assert ask(Q.even(re(x)), ~Q.even(x)) is None
    assert ask(Q.even(im(x)), Q.even(x)) is True
    assert ask(Q.even(im(x)), Q.real(x)) is True
    assert ask(Q.even((-1) ** n), Q.integer(n)) is False
    assert ask(Q.even(k ** 2), Q.even(k)) is True
    assert ask(Q.even(n ** 2), Q.odd(n)) is False
    assert ask(Q.even(2 ** k), Q.even(k)) is None
    assert ask(Q.even(x ** 2)) is None
    assert ask(Q.even(k ** m), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.even(n ** m), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is False
    assert ask(Q.even(k ** p), Q.even(k) & Q.integer(p) & Q.positive(p)) is True
    assert ask(Q.even(n ** p), Q.odd(n) & Q.integer(p) & Q.positive(p)) is False
    assert ask(Q.even(m ** k), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.even(p ** k), Q.even(k) & Q.integer(p) & Q.positive(p)) is None
    assert ask(Q.even(m ** n), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.even(p ** n), Q.odd(n) & Q.integer(p) & Q.positive(p)) is None
    assert ask(Q.even(k ** x), Q.even(k)) is None
    assert ask(Q.even(n ** x), Q.odd(n)) is None
    assert ask(Q.even(x * y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.even(x * x), Q.integer(x)) is None
    assert ask(Q.even(x * (x + y)), Q.integer(x) & Q.odd(y)) is True
    assert ask(Q.even(x * (x + y)), Q.integer(x) & Q.even(y)) is None