import filecmp
import pytest
from ase.build import molecule
from ase.io import read, write
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
def atoms_equal(atoms1, atoms2):
    return compare_atoms(atoms1, atoms2, tol=1e-08) == []