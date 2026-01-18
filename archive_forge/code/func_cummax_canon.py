from cvxpy.expressions.variable import Variable
def cummax_canon(expr, args):
    """Cumulative max.
    """
    X = args[0]
    axis = expr.axis
    Y = Variable(expr.shape)
    constr = [X <= Y]
    if axis == 0:
        if expr.shape[0] == 1:
            return (X, [])
        else:
            constr += [Y[:-1] <= Y[1:]]
    elif expr.shape[1] == 1:
        return (X, [])
    else:
        constr += [Y[:, :-1] <= Y[:, 1:]]
    return (Y, constr)