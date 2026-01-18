import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def get_distances_derivatives(v0, cell=None, pbc=None):
    """Get derivatives of distances for all vectors in v0 w.r.t. Cartesian
    coordinates in Angstrom.

    Set cell and pbc to use the minimum image convention.

    There is a singularity for distances -> 0 for which a ZeroDivisionError is
    raised.
    Derivative output format: [[dx_a0, dy_a0, dz_a0], [dx_a1, dy_a1, dz_a1]].
    """
    (v0,), (dists,) = conditional_find_mic([v0], cell, pbc)
    if (dists <= 0.0).any():
        raise ZeroDivisionError('Singularity for distance derivative')
    derivs_d0 = np.einsum('i,ij->ij', -1.0 / dists, v0)
    derivs_d1 = -derivs_d0
    derivs = np.stack((derivs_d0, derivs_d1), axis=1)
    return derivs