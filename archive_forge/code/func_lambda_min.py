from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.expressions.expression import Expression
def lambda_min(X):
    """ Minimum eigenvalue; :math:`\\lambda_{\\min}(A)`.
    """
    X = Expression.cast_to_const(X)
    return -lambda_max(-X)