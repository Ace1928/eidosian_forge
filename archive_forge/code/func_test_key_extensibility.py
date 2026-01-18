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
def test_key_extensibility():
    """test that you can add keys to the ask system at runtime"""
    raises(AttributeError, lambda: ask(Q.my_key(x)))

    class MyAskHandler(AskHandler):

        @staticmethod
        def Symbol(expr, assumptions):
            return True
    try:
        with warns_deprecated_sympy():
            register_handler('my_key', MyAskHandler)
        with warns_deprecated_sympy():
            assert ask(Q.my_key(x)) is True
        with warns_deprecated_sympy():
            assert ask(Q.my_key(x + 1)) is None
    finally:
        with warns_deprecated_sympy():
            remove_handler('my_key', MyAskHandler)
        del Q.my_key
    raises(AttributeError, lambda: ask(Q.my_key(x)))

    class MyPredicate(Predicate):
        pass
    try:
        Q.my_key = MyPredicate()

        @Q.my_key.register(Symbol)
        def _(expr, assumptions):
            return True
        assert ask(Q.my_key(x)) is True
        assert ask(Q.my_key(x + 1)) is None
    finally:
        del Q.my_key
    raises(AttributeError, lambda: ask(Q.my_key(x)))