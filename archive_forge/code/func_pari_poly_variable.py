from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def pari_poly_variable(variable_name):
    """
    Ensures that PARI has the requested polynomial variable defined.
    If "variable_name" is already defined in PARI as something else,
    an exception is raised.

    >>> val = 3*pari_poly_variable('silly')**2; val
    3*silly^2
    >>> ten = pari('silly = 10')
    >>> pari_poly_variable('silly')
    Traceback (most recent call last):
    ...
    RuntimeError: In PARI, "silly" is already defined
    """
    p = pari(variable_name)
    success = p.type() == 't_POL' and p.variables() == [p]
    if not success:
        raise RuntimeError('In PARI, "%s" is already defined' % variable_name)
    return p