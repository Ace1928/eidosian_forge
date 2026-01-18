import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def gm_constrs(t, x_list, p):
    """ Form the internal CXVPY constraints to form the weighted geometric mean t <= x^p.

    t <= x[0]^p[0] * x[1]^p[1] * ... * x[n]^p[n]

    where x and t can either be scalar or matrix variables.

    Parameters
    ----------

    t : cvx.Variable
        The epigraph variable

    x_list : list of cvx.Variable objects
        The vector of input variables. Must be the same length as ``p``.

    p : list or tuple of ``int`` and ``Fraction`` objects
        The powers vector. powers must be nonnegative and sum to *exactly* 1.
        Must be the same length as ``x``.

    Returns
    -------
    constr : list
        list of constraints involving elements of x (and possibly t) to form the geometric mean.

    """
    assert is_weight(p)
    w = dyad_completion(p)
    tree = decompose(w)
    d = defaultdict(lambda: Variable(t.shape))
    d[w] = t
    long_w = len(w) - len(x_list)
    if long_w > 0:
        x_list += [t] * long_w
    assert len(x_list) == len(w)
    for i, (p, v) in enumerate(zip(w, x_list)):
        if p > 0:
            tmp = [0] * len(w)
            tmp[i] = 1
            d[tuple(tmp)] = v
    constraints = []
    for elem, children in tree.items():
        if 1 not in elem:
            constraints += [gm(d[elem], d[children[0]], d[children[1]])]
    return constraints