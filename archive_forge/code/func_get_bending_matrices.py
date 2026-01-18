import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def get_bending_matrices(self, J, ecc):
    """
        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities

        Returns
        -------
        Returns the element K matrices for bending energy expressed in
        GLOBAL nodal coordinates.
        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]
        tri_J is needed to rotate dofs from local basis to global basis
        """
    n = np.size(ecc, 0)
    J1 = self.J0_to_J1 @ J
    J2 = self.J0_to_J2 @ J
    DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)
    DOF_rot[:, 0, 0] = 1
    DOF_rot[:, 3, 3] = 1
    DOF_rot[:, 6, 6] = 1
    DOF_rot[:, 1:3, 1:3] = J
    DOF_rot[:, 4:6, 4:6] = J1
    DOF_rot[:, 7:9, 7:9] = J2
    H_rot, area = self.get_Hrot_from_J(J, return_area=True)
    K = np.zeros([n, 9, 9], dtype=np.float64)
    weights = self.gauss_w
    pts = self.gauss_pts
    for igauss in range(self.n_gauss):
        alpha = np.tile(pts[igauss, :], n).reshape(n, 3)
        alpha = np.expand_dims(alpha, 2)
        weight = weights[igauss]
        d2Skdksi2 = self.get_d2Sidksij2(alpha, ecc)
        d2Skdx2 = d2Skdksi2 @ H_rot
        K += weight * (d2Skdx2 @ self.E @ _transpose_vectorized(d2Skdx2))
    K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot
    return _scalar_vectorized(area, K)