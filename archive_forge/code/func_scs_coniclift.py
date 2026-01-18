import numpy as np
from scipy import sparse
from cvxpy.atoms.suppfunc import SuppFuncAtom
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CONVEX_ATTRIBUTES
def scs_coniclift(x, constraints):
    """
    Return (A, b, K) so that
        {x : x satisfies constraints}
    can be written as
        {x : exists y where A @ [x; y] + b in K}.

    Parameters
    ----------
    x: cvxpy.Variable
    constraints: list of cvxpy.constraints.constraint.Constraint
        Each Constraint object must be DCP-compatible.

    Notes
    -----
    This function DOES NOT work when ``x`` has attributes, like ``PSD=True``,
    ``diag=True``, ``symmetric=True``, etc...
    """
    from cvxpy.atoms.affine.sum import sum
    from cvxpy.problems.objective import Minimize
    from cvxpy.problems.problem import Problem
    prob = Problem(Minimize(sum(x)), constraints)
    data, chain, invdata = prob.get_problem_data(solver='SCS')
    inv = invdata[-2]
    x_offset = inv.var_offsets[x.id]
    x_indices = np.arange(x_offset, x_offset + x.size)
    A = data['A']
    x_selector = np.zeros(shape=(A.shape[1],), dtype=bool)
    x_selector[x_indices] = True
    A_x = A[:, x_selector]
    A_other = A[:, ~x_selector]
    A = -sparse.hstack([A_x, A_other])
    b = data['b']
    K = data['dims']
    return (A, b, K)