import numpy as np
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.solution import Solution
def pow_nd_canon(con, args):
    """
    con : PowConeND
        We can extract metadata from this.
        For example, con.alpha and con.axis.
    args : tuple of length two
        W,z = args[0], args[1]
    """
    alpha, axis, _ = con.get_data()
    alpha = alpha.value
    W, z = args
    if axis == 1:
        W = W.T
        alpha = alpha.T
    if W.ndim == 1:
        W = reshape(W, (W.size, 1))
        alpha = np.reshape(alpha, (W.size, 1))
    n, k = W.shape
    if n == 2:
        can_con = PowCone3D(W[0, :], W[1, :], z, alpha[0, :])
    else:
        T = Variable(shape=(n - 2, k))
        x_3d, y_3d, z_3d, alpha_3d = ([], [], [], [])
        for j in range(k):
            x_3d.append(W[:-1, j])
            y_3d.append(T[:, j])
            y_3d.append(W[n - 1, j])
            z_3d.append(z[j])
            z_3d.append(T[:, j])
            r_nums = alpha[:, j]
            r_dens = np.cumsum(r_nums[::-1])[::-1]
            r = r_nums / r_dens
            alpha_3d.append(r[:n - 1])
        x_3d = hstack(x_3d)
        y_3d = hstack(y_3d)
        z_3d = hstack(z_3d)
        alpha_p3d = hstack(alpha_3d)
        can_con = PowCone3D(x_3d, y_3d, z_3d, alpha_p3d)
    return (can_con, [])