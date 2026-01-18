import random
from mpmath import *
from mpmath.libmp import *
def test_eval_repr_invariant():
    """Test that eval(repr(x)) == x"""
    random.seed(123)
    for dps in [10, 15, 20, 50, 100]:
        mp.dps = dps
        for i in range(1000):
            a = mpf(random.random()) ** 0.5 * 10 ** random.randint(-100, 100)
            assert eval(repr(a)) == a
    mp.dps = 15