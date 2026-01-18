import numpy as np
from cvxpy.lin_ops.tree_mat import mul, sum_dicts, tmul
def constr_unpack(constraints, vector):
    """Unpacks a vector into a list of values for constraints.
    """
    values = []
    offset = 0
    for constr in constraints:
        rows, cols = constr.size
        val = np.zeros((rows, cols))
        for col in range(cols):
            val[:, col] = vector[offset:offset + rows]
            offset += rows
        values.append(val)
    return values