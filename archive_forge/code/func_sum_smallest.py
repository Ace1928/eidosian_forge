from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.expressions.expression import Expression
def sum_smallest(x, k):
    """Sum of the smallest k values.
    """
    x = Expression.cast_to_const(x)
    return -sum_largest(-x, k)