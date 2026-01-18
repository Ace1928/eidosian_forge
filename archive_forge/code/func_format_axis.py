from typing import Tuple
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
def format_axis(t, X, axis):
    """Formats all the row/column cones for the solver.

    Parameters
    ----------
        t: The scalar part of the second-order constraint.
        X: A matrix whose rows/columns are each a cone.
        axis: Slice by column 0 or row 1.

    Returns
    -------
    list
        A list of LinLeqConstr that represent all the elementwise cones.
    """
    if axis == 1:
        X = lu.transpose(X)
    cone_size = 1 + X.shape[0]
    terms = []
    mat_shape = (cone_size, 1)
    t_mat = sp.csc_matrix(([1.0], ([0], [0])), mat_shape)
    t_mat = lu.create_const(t_mat, mat_shape, sparse=True)
    t_vec = t
    if not t.shape:
        t_vec = lu.reshape(t, (1, 1))
    else:
        t_vec = lu.reshape(t, (1, t.shape[0]))
    mul_shape = (cone_size, t_vec.shape[1])
    terms += [lu.mul_expr(t_mat, t_vec, mul_shape)]
    if len(X.shape) == 1:
        X = lu.reshape(X, (X.shape[0], 1))
    mat_shape = (cone_size, X.shape[0])
    val_arr = (cone_size - 1) * [1.0]
    row_arr = range(1, cone_size)
    col_arr = range(cone_size - 1)
    X_mat = sp.csc_matrix((val_arr, (row_arr, col_arr)), mat_shape)
    X_mat = lu.create_const(X_mat, mat_shape, sparse=True)
    mul_shape = (cone_size, X.shape[1])
    terms += [lu.mul_expr(X_mat, X, mul_shape)]
    return [lu.create_geq(lu.sum_expr(terms))]