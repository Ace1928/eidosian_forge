from typing import List, Tuple, Union
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.constraints.constraint import Constraint
from cvxpy.utilities.power_tools import pow_high, pow_mid, pow_neg
def pnorm(x, p: Union[int, str]=2, axis=None, keepdims: bool=False, max_denom: int=1024):
    """Factory function for a mathematical p-norm.

    Parameters
    ----------
    p : numeric type or string
       The type of norm to construct; set this to np.inf or 'inf' to
       construct an infinity norm.

    Returns
    -------
    Atom
       A norm1, norm_inf, or Pnorm object.
    """
    if p == 1:
        return norm1(x, axis=axis, keepdims=keepdims)
    elif p in [np.inf, 'inf', 'Inf']:
        return norm_inf(x, axis=axis, keepdims=keepdims)
    else:
        return Pnorm(x, p=p, axis=axis, keepdims=keepdims, max_denom=max_denom)