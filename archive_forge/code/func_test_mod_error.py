from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_mod_error():
    raises(ZeroDivisionError, lambda: mod(2, 0))