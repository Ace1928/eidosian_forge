from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=None):
    """Create three-layer 2D materials with hexagonal structure.

    For metal dichalcogenites, etc.

    The kind argument accepts '2H', which gives a mirror plane symmetry
    and '1T', which gives an inversion symmetry."""
    if kind == '2H':
        basis = [(0, 0, 0), (2 / 3, 1 / 3, 0.5 * thickness), (2 / 3, 1 / 3, -0.5 * thickness)]
    elif kind == '1T':
        basis = [(0, 0, 0), (2 / 3, 1 / 3, 0.5 * thickness), (1 / 3, 2 / 3, -0.5 * thickness)]
    else:
        raise ValueError('Structure not recognized:', kind)
    cell = [[a, 0, 0], [-a / 2, a * 3 ** 0.5 / 2, 0], [0, 0, 0]]
    atoms = Atoms(formula, cell=cell, pbc=(1, 1, 0))
    atoms.set_scaled_positions(basis)
    if vacuum is not None:
        atoms.center(vacuum, axis=2)
    atoms = atoms.repeat(size)
    return atoms