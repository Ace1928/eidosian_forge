import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def get_angles_derivatives(v0, v1, cell=None, pbc=None):
    """Get derivatives of angles formed by two lists of vectors (v0, v1) w.r.t.
    Cartesian coordinates in degrees.

    Set a cell and pbc to enable minimum image
    convention, otherwise derivatives of angles are taken as-is.

    There is a singularity in the derivatives for sin(angle) -> 0 for which
    a ZeroDivisionError is raised.

    Derivative output format: [[dx_a0, dy_a0, dz_a0], [...], [..., dz_a2].
    """
    (v0, v1), (nv0, nv1) = conditional_find_mic([v0, v1], cell, pbc)
    angles = np.radians(get_angles(v0, v1, cell=cell, pbc=pbc))
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    if (sin_angles == 0.0).any():
        raise ZeroDivisionError('Singularity for angle derivative')
    product = nv0 * nv1
    deriv_d0 = -(v1 / product[:, np.newaxis] - np.einsum('ij,i->ij', v0, cos_angles / nv0 ** 2)) / sin_angles[:, np.newaxis]
    deriv_d2 = -(v0 / product[:, np.newaxis] - np.einsum('ij,i->ij', v1, cos_angles / nv1 ** 2)) / sin_angles[:, np.newaxis]
    deriv_d1 = -(deriv_d0 + deriv_d2)
    derivs = np.stack((deriv_d0, deriv_d1, deriv_d2), axis=1)
    return np.degrees(derivs)