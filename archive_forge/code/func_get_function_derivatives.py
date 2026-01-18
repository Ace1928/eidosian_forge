import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def get_function_derivatives(self, alpha, J, ecc, dofs):
    """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices of
        barycentric coordinates)
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
        eccentricities)
        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
        in global coordinates at locations alpha, as a column-matrices of
        shape (N x 2 x 1).
        """
    subtri = np.argmin(alpha, axis=1)[:, 0]
    ksi = _roll_vectorized(alpha, -subtri, axis=0)
    E = _roll_vectorized(ecc, -subtri, axis=0)
    x = ksi[:, 0, 0]
    y = ksi[:, 1, 0]
    z = ksi[:, 2, 0]
    x_sq = x * x
    y_sq = y * y
    z_sq = z * z
    dV = _to_matrix_vectorized([[-3.0 * x_sq, -3.0 * x_sq], [3.0 * y_sq, 0.0], [0.0, 3.0 * z_sq], [-2.0 * x * z, -2.0 * x * z + x_sq], [-2.0 * x * y + x_sq, -2.0 * x * y], [2.0 * x * y - y_sq, -y_sq], [2.0 * y * z, y_sq], [z_sq, 2.0 * y * z], [-z_sq, 2.0 * x * z - z_sq], [x * z - y * z, x * y - y * z]])
    dV = dV @ _extract_submatrices(self.rotate_dV, subtri, block_size=2, axis=0)
    prod = self.M @ dV
    prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
    prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
    prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
    dsdksi = _roll_vectorized(prod, 3 * subtri, axis=0)
    dfdksi = dofs @ dsdksi
    J_inv = _safe_inv22_vectorized(J)
    dfdx = J_inv @ _transpose_vectorized(dfdksi)
    return dfdx