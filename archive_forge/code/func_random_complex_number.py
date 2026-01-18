import ``rng`` and access the method directly. For example, to capture the
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
import random as _random
def random_complex_number(a=2, b=-1, c=3, d=1, rational=False, tolerance=None):
    """
    Return a random complex number.

    To reduce chance of hitting branch cuts or anything, we guarantee
    b <= Im z <= d, a <= Re z <= c

    When rational is True, a rational approximation to a random number
    is obtained within specified tolerance, if any.
    """
    from sympy.core.numbers import I
    from sympy.simplify.simplify import nsimplify
    A, B = (uniform(a, c), uniform(b, d))
    if not rational:
        return A + I * B
    return nsimplify(A, rational=True, tolerance=tolerance) + I * nsimplify(B, rational=True, tolerance=tolerance)