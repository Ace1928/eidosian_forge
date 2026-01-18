from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def test_FactRules_parse2():
    raises(ValueError, lambda: FactRules('a -> !a'))