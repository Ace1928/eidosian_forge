from ..sage_helper import _within_sage, sage_method, SageNotAvailable
def eval_factor_yEqn(p):
    """
        Evaluation method for the factors of the equation factored over the
        number field. We take the factor and turn it into a polynomial in
        Q[x][y]. We then put in the given intervals for x and y.
        """
    lift = p.map_coefficients(lambda c: c.lift('x'), Rx)
    return lift.substitute(x=x_val, y=y_val)