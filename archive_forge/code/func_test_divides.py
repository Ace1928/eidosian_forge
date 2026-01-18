from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_divides():
    assert divides(5, 2) is False
    assert divides(10, 5) is True
    assert divides(0, 0) is True
    assert divides(5, 0) is False