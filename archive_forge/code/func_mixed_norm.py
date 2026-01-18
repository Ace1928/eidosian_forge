from typing import Union
from cvxpy.atoms.norm import norm
from cvxpy.expressions.expression import Expression
def mixed_norm(X, p: Union[int, str]=2, q: Union[int, str]=1):
    """Lp,q norm; :math:`(\\sum_k (\\sum_l \\lvert x_{k,l} \\rvert^p)^{q/p})^{1/q}`.

    Parameters
    ----------
    X : Expression or numeric constant
        The matrix to take the l_{p,q} norm of.
    p : int or str, optional
        The type of inner norm.
    q : int or str, optional
        The type of outer norm.

    Returns
    -------
    Expression
        An Expression representing the mixed norm.
    """
    X = Expression.cast_to_const(X)
    vecnorms = norm(X, p, axis=1)
    return norm(vecnorms, q)