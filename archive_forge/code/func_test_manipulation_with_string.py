import pytest
from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols
def test_manipulation_with_string():
    atoms = molecule('H2O')
    atoms.symbols = 'Au2Ag'
    print(atoms.symbols)
    assert (atoms.symbols == 'Au2Ag').all()