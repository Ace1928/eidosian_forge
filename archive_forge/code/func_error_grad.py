import scipy.sparse as sp
def error_grad(expr):
    """Returns a gradient of all None.

    Args:
        expr: An expression.

    Returns:
        A map of variable value to None.
    """
    return {var: None for var in expr.variables()}