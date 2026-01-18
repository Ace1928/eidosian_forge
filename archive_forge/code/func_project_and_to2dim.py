from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def project_and_to2dim(self, pps, plane_center):
    """
        Projects the list of points pps to the plane and changes the basis from 3D to the 2D basis of the plane

        Args:
            pps: List of points to be projected

        Returns:
            :raise:
        """
    proj = self.projectionpoints(pps)
    [u1, u2, u3] = self.orthonormal_vectors()
    PP = np.array([[u1[0], u2[0], u3[0]], [u1[1], u2[1], u3[1]], [u1[2], u2[2], u3[2]]])
    xypps = []
    for pp in proj:
        xyzpp = np.dot(pp, PP)
        xypps.append(xyzpp[0:2])
    if str(plane_center) == 'mean':
        mean = np.zeros(2, float)
        for pp in xypps:
            mean += pp
        mean /= len(xypps)
        xypps = [pp - mean for pp in xypps]
    elif plane_center is not None:
        projected_plane_center = self.projectionpoints([plane_center])[0]
        xy_projected_plane_center = np.dot(projected_plane_center, PP)[0:2]
        xypps = [pp - xy_projected_plane_center for pp in xypps]
    return xypps