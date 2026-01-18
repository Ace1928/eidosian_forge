import numpy as np
import scipy
from . import _voronoi
from scipy.spatial import cKDTree
def calculate_solid_angles(R):
    """Calculates the solid angles of plane triangles. Implements the method of
    Van Oosterom and Strackee [VanOosterom]_ with some modifications. Assumes
    that input points have unit norm."""
    numerator = np.linalg.det(R)
    denominator = 1 + (np.einsum('ij,ij->i', R[:, 0], R[:, 1]) + np.einsum('ij,ij->i', R[:, 1], R[:, 2]) + np.einsum('ij,ij->i', R[:, 2], R[:, 0]))
    return np.abs(2 * np.arctan2(numerator, denominator))