import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def adjust_forces(self, atoms, forces):
    forces[:] = symmetrize_rank1(atoms.get_cell(), atoms.cell.reciprocal().T, forces, self.rotations, self.translations, self.symm_map)