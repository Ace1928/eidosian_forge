import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def permute_axes(atoms, permutation):
    """Permute axes of unit cell and atom positions. Considers only cell and
    atomic positions. Other vector quantities such as momenta are not
    modified."""
    assert (np.sort(permutation) == np.arange(3)).all()
    permuted = atoms.copy()
    scaled = permuted.get_scaled_positions()
    permuted.set_cell(permuted.cell.permute_axes(permutation), scale_atoms=False)
    permuted.set_scaled_positions(scaled[:, permutation])
    permuted.set_pbc(permuted.pbc[permutation])
    return permuted