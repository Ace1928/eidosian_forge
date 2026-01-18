from math import pi
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, kpts2ndarray
from ase.units import Bohr, Ha
def numeric_force(atoms, a, i, d=0.001):
    """Compute numeric force on atom with index a, Cartesian component i,
    with finite step of size d
    """
    p0 = atoms.get_positions()
    p = p0.copy()
    p[a, i] += d
    atoms.set_positions(p, apply_constraint=False)
    eplus = atoms.get_potential_energy()
    p[a, i] -= 2 * d
    atoms.set_positions(p, apply_constraint=False)
    eminus = atoms.get_potential_energy()
    atoms.set_positions(p0, apply_constraint=False)
    return (eminus - eplus) / (2 * d)