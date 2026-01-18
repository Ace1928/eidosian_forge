from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_nextprime():
    assert nextprime(-3) == 2
    assert nextprime(5) == 7
    assert nextprime(9) == 11