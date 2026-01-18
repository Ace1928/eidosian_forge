import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def get_dihedrals_derivatives(v0, v1, v2, cell=None, pbc=None):
    """Get derivatives of dihedrals formed by three lists of vectors
    (v0, v1, v2) w.r.t Cartesian coordinates in degrees.

    Set a cell and pbc to enable minimum image
    convention, otherwise dihedrals are taken as-is.

    Derivative output format: [[dx_a0, dy_a0, dz_a0], ..., [..., dz_a3]].
    """
    (v0, v1, v2), (nv0, nv1, nv2) = conditional_find_mic([v0, v1, v2], cell, pbc)
    v0 /= nv0[:, np.newaxis]
    v1 /= nv1[:, np.newaxis]
    v2 /= nv2[:, np.newaxis]
    normal_v01 = np.cross(v0, v1, axis=1)
    normal_v12 = np.cross(v1, v2, axis=1)
    cos_psi01 = np.einsum('ij,ij->i', v0, v1)
    sin_psi01 = np.sin(np.arccos(cos_psi01))
    cos_psi12 = np.einsum('ij,ij->i', v1, v2)
    sin_psi12 = np.sin(np.arccos(cos_psi12))
    if (sin_psi01 == 0.0).any() or (sin_psi12 == 0.0).any():
        raise ZeroDivisionError('Undefined derivative for undefined dihedral')
    deriv_d0 = -normal_v01 / (nv0 * sin_psi01 ** 2)[:, np.newaxis]
    deriv_d3 = normal_v12 / (nv2 * sin_psi12 ** 2)[:, np.newaxis]
    deriv_d1 = ((nv1 + nv0 * cos_psi01) / nv1)[:, np.newaxis] * -deriv_d0 + (cos_psi12 * nv2 / nv1)[:, np.newaxis] * deriv_d3
    deriv_d2 = -((nv1 + nv2 * cos_psi12) / nv1)[:, np.newaxis] * deriv_d3 - (cos_psi01 * nv0 / nv1)[:, np.newaxis] * -deriv_d0
    derivs = np.stack((deriv_d0, deriv_d1, deriv_d2, deriv_d3), axis=1)
    return np.degrees(derivs)