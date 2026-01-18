import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.lj import LennardJones
from ase.spacegroup.symmetrize import FixSymmetry, check_symmetry, is_subgroup
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.constraints import UnitCellFilter, ExpCellFilter
def perturb(atoms, pos0, at_i, dpos):
    positions = pos0.copy()
    positions[at_i] += dpos
    atoms.set_positions(positions)
    new_p = atoms.get_positions()
    return pos0[at_i] - new_p[at_i]