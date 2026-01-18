from . import matrix
from deprecated import deprecated
import numbers
import warnings
def check_greater(x, **params):
    """Check that parameters are greater than x as expected

    Parameters
    ----------

    x : excepted boundary
        Checks not run if parameters are greater than x

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Number) or params[p] <= x:
            raise ValueError('Expected {} > {}, got {}'.format(p, x, params[p]))