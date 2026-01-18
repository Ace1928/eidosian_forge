import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def general_find_mic(v, cell, pbc=True):
    """Finds the minimum-image representation of vector(s) v. Using the
    Minkowski reduction the algorithm is relatively slow but safe for any cell.
    """
    cell = complete_cell(cell)
    rcell, _ = minkowski_reduce(cell, pbc=pbc)
    positions = wrap_positions(v, rcell, pbc=pbc, eps=0)
    ranges = [np.arange(-1 * p, p + 1) for p in pbc]
    hkls = np.array([(0, 0, 0)] + list(itertools.product(*ranges)))
    vrvecs = hkls @ rcell
    x = positions + vrvecs[:, None]
    lengths = np.linalg.norm(x, axis=2)
    indices = np.argmin(lengths, axis=0)
    vmin = x[indices, np.arange(len(positions)), :]
    vlen = lengths[indices, np.arange(len(positions))]
    return (vmin, vlen)