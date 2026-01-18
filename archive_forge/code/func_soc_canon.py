from cvxpy.atoms import reshape, vstack
from cvxpy.constraints import SOC
from cvxpy.expressions.variable import Variable
def soc_canon(expr, real_args, imag_args, real2imag):
    if real_args[1] is None:
        output = [SOC(real_args[0], imag_args[1], axis=expr.axis, constr_id=real2imag[expr.id])]
    elif imag_args[1] is None:
        output = [SOC(real_args[0], real_args[1], axis=expr.axis, constr_id=expr.id)]
    else:
        orig_shape = real_args[1].shape
        real = real_args[1].flatten()
        imag = imag_args[1].flatten()
        flat_X = Variable(real.shape)
        inner_SOC = SOC(flat_X, vstack([real, imag]), axis=0)
        real_X = reshape(flat_X, orig_shape)
        outer_SOC = SOC(real_args[0], real_X, axis=expr.axis, constr_id=expr.id)
        output = [inner_SOC, outer_SOC]
    return (output, None)