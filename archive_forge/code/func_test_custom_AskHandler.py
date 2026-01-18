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
def test_custom_AskHandler():
    from sympy.logic.boolalg import conjuncts

    class MersenneHandler(AskHandler):

        @staticmethod
        def Integer(expr, assumptions):
            if ask(Q.integer(log(expr + 1, 2))):
                return True

        @staticmethod
        def Symbol(expr, assumptions):
            if expr in conjuncts(assumptions):
                return True
    try:
        with warns_deprecated_sympy():
            register_handler('mersenne', MersenneHandler)
        n = Symbol('n', integer=True)
        with warns_deprecated_sympy():
            assert ask(Q.mersenne(7))
        with warns_deprecated_sympy():
            assert ask(Q.mersenne(n), Q.mersenne(n))
    finally:
        del Q.mersenne

    class MersennePredicate(Predicate):
        pass
    try:
        Q.mersenne = MersennePredicate()

        @Q.mersenne.register(Integer)
        def _(expr, assumptions):
            if ask(Q.integer(log(expr + 1, 2))):
                return True

        @Q.mersenne.register(Symbol)
        def _(expr, assumptions):
            if expr in conjuncts(assumptions):
                return True
        assert ask(Q.mersenne(7))
        assert ask(Q.mersenne(n), Q.mersenne(n))
    finally:
        del Q.mersenne