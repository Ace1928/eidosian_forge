import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def create_Poly(parameter_name, reciprocal=False, shift=None, name=None):
    """
    Examples
    --------
    >>> Poly = create_Poly('x')
    >>> p1 = Poly([3, 4, 5])
    >>> p1({'x': 7}) == 3 + 4*7 + 5*49
    True
    >>> RPoly = create_Poly('T', reciprocal=True)
    >>> p2 = RPoly([64, 32, 16, 8])
    >>> p2({'T': 2}) == 64 + 16 + 4 + 1
    True
    >>> SPoly = create_Poly('z', shift=True)
    >>> p3 = SPoly([7, 2, 3, 5], unique_keys=('z0',))
    >>> p3({'z': 9}) == 2 + 3*(9-7) + 5*(9-7)**2
    True
    >>> p3({'z': 9, 'z0': 6}) == 2 + 3*(9-6) + 5*(9-6)**2
    True

    """
    if shift is True:
        shift = 'shift'

    def _poly(args, x, backend=math, **kwargs):
        if shift is None:
            coeffs = args
            x0 = x
        else:
            coeffs = args[1:]
            x_shift = args[0]
            x0 = x - x_shift
        cur = 1
        res = None
        for coeff in coeffs:
            if res is None:
                res = coeff * cur
            else:
                res += coeff * cur
            if reciprocal:
                cur /= x0
            else:
                cur *= x0
        return res
    if shift is None:
        argument_names = None
    else:
        argument_names = (shift, Ellipsis)
    if name is not None:
        _poly.__name__ = name
    return Expr.from_callback(_poly, parameter_keys=(parameter_name,), argument_names=argument_names)