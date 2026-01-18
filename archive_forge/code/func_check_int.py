from . import matrix
from deprecated import deprecated
import numbers
import warnings
def check_int(**params):
    """Check that parameters are integers as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Integral):
            raise ValueError('Expected {} integer, got {}'.format(p, params[p]))