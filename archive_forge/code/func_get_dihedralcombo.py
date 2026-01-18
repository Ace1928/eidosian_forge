from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def get_dihedralcombo(atoms, dihedralcombo_def):
    return sum([defin[4] * atoms.get_dihedral(*defin[0:4]) for defin in dihedralcombo_def])