from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_fibonacci_error():
    raises(NotImplementedError, lambda: fibonacci(-3))