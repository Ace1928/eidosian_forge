from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_binomial_error():
    raises(ArithmeticError, lambda: binomial(5, -1))