import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def get_Hrot_from_J(self, J, return_area=False):
    """
        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)

        Returns
        -------
        Returns H_rot used to rotate Hessian from local basis of first apex,
        to global coordinates.
        if *return_area* is True, returns also the triangle area (0.5*det(J))
        """
    J_inv = _safe_inv22_vectorized(J)
    Ji00 = J_inv[:, 0, 0]
    Ji11 = J_inv[:, 1, 1]
    Ji10 = J_inv[:, 1, 0]
    Ji01 = J_inv[:, 0, 1]
    H_rot = _to_matrix_vectorized([[Ji00 * Ji00, Ji10 * Ji10, Ji00 * Ji10], [Ji01 * Ji01, Ji11 * Ji11, Ji01 * Ji11], [2 * Ji00 * Ji01, 2 * Ji11 * Ji10, Ji00 * Ji11 + Ji10 * Ji01]])
    if not return_area:
        return H_rot
    else:
        area = 0.5 * (J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
        return (H_rot, area)