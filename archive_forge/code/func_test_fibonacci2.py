from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_fibonacci2():
    assert fibonacci2(0) == [0, 1]
    assert fibonacci2(5) == [5, 3]