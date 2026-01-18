import numpy as np
from numpy import pi, sin, cos, arccos, sqrt, dot
from numpy.linalg import norm
from ase.cell import Cell  # noqa
def complete_cell(cell):
    """Calculate complete cell with missing lattice vectors.

    Returns a new 3x3 ndarray.
    """
    cell = np.array(cell, dtype=float)
    missing = np.nonzero(~cell.any(axis=1))[0]
    if len(missing) == 3:
        cell.flat[::4] = 1.0
    if len(missing) == 2:
        V, s, WT = np.linalg.svd(cell.T)
        sf = [s[0], 1, 1]
        cell = (V @ np.diag(sf) @ WT).T
        if np.sign(np.linalg.det(cell)) < 0:
            cell[missing[0]] = -cell[missing[0]]
    elif len(missing) == 1:
        i = missing[0]
        cell[i] = np.cross(cell[i - 2], cell[i - 1])
        cell[i] /= np.linalg.norm(cell[i])
    return cell