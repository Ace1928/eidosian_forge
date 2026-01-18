import itertools
import numpy as np
from ase.utils import pbc2pbc
from ase.cell import Cell
def is_minkowski_reduced(cell, pbc=True):
    """Tests if a cell is Minkowski-reduced.

    Parameters:

    cell: array
        The lattice basis to test (in row-vector format).
    pbc: array, optional
        The periodic boundary conditions of the cell (Default `True`).
        If `pbc` is provided, only periodic cell vectors are tested.

    Returns:

    is_reduced: bool
        True if cell is Minkowski-reduced, False otherwise.
    """
    'These conditions are due to Minkowski, but a nice description in English\n    can be found in the thesis of Carine Jaber: "Algorithmic approaches to\n    Siegel\'s fundamental domain", https://www.theses.fr/2017UBFCK006.pdf\n    This is also good background reading for Minkowski reduction.\n\n    0D and 1D cells are trivially reduced. For 2D cells, the conditions which\n    an already-reduced basis fulfil are:\n    |b1| ≤ |b2|\n    |b2| ≤ |b1 - b2|\n    |b2| ≤ |b1 + b2|\n\n    For 3D cells, the conditions which an already-reduced basis fulfil are:\n    |b1| ≤ |b2| ≤ |b3|\n\n    |b1 + b2|      ≥ |b2|\n    |b1 + b3|      ≥ |b3|\n    |b2 + b3|      ≥ |b3|\n    |b1 - b2|      ≥ |b2|\n    |b1 - b3|      ≥ |b3|\n    |b2 - b3|      ≥ |b3|\n    |b1 + b2 + b3| ≥ |b3|\n    |b1 - b2 + b3| ≥ |b3|\n    |b1 + b2 - b3| ≥ |b3|\n    |b1 - b2 - b3| ≥ |b3|\n    '
    pbc = pbc2pbc(pbc)
    dim = pbc.sum()
    if dim <= 1:
        return True
    if dim == 2:
        cell = cell.copy()
        cell[np.argmin(pbc)] = 0
        norms = np.linalg.norm(cell, axis=1)
        cell = cell[np.argsort(norms)[[1, 2, 0]]]
        A = [[0, 1, 0], [1, -1, 0], [1, 1, 0]]
        lhs = np.linalg.norm(A @ cell, axis=1)
        norms = np.linalg.norm(cell, axis=1)
        rhs = norms[[0, 1, 1]]
    else:
        A = [[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1], [1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1]]
        lhs = np.linalg.norm(A @ cell, axis=1)
        norms = np.linalg.norm(cell, axis=1)
        rhs = norms[[0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2]]
    return (lhs >= rhs - TOL).all()