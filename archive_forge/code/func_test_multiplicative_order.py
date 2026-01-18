from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_multiplicative_order():
    assert multiplicative_order(2, 21) == 6
    assert multiplicative_order(5, 10) is None