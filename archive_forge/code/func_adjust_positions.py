import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def adjust_positions(self, atoms, new):
    if not self.do_adjust_positions:
        return
    step = new - atoms.positions
    symmetrized_step = symmetrize_rank1(atoms.get_cell(), atoms.cell.reciprocal().T, step, self.rotations, self.translations, self.symm_map)
    new[:] = atoms.positions + symmetrized_step