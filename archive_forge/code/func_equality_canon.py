import numpy as np
from cvxpy.constraints.zero import Equality, Zero
from cvxpy.expressions.constants import Constant
def equality_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is None and imag_args[1] is None:
        return ([expr.copy(real_args)], None)
    for i in range(len(imag_args)):
        if imag_args[i] is None:
            imag_args[i] = Constant(np.zeros(real_args[i].shape))
    imag_cons = [Equality(imag_args[0], imag_args[1], constr_id=real2imag[expr.id])]
    if real_args[0] is None and real_args[1] is None:
        return (None, imag_cons)
    else:
        for i in range(len(real_args)):
            if real_args[i] is None:
                real_args[i] = Constant(np.zeros(imag_args[i].shape))
        return ([expr.copy(real_args)], imag_cons)