from sympy.core.symbol import Symbol
from sympy.plotting.intervalmath import interval
from sympy.plotting.intervalmath.interval_membership import intervalMembership
from sympy.plotting.experimental_lambdify import experimental_lambdify
from sympy.testing.pytest import raises
def test_boolean_errors():
    a = intervalMembership(True, True)
    raises(ValueError, lambda: a & 1)
    raises(ValueError, lambda: a | 1)
    raises(ValueError, lambda: a ^ 1)