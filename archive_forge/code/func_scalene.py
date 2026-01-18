from cvxpy.atoms.elementwise.neg import neg
from cvxpy.atoms.elementwise.pos import pos
def scalene(x, alpha, beta):
    """ Alias for ``alpha*pos(x) + beta*neg(x)``.
    """
    return alpha * pos(x) + beta * neg(x)