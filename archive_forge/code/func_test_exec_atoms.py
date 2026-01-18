import pytest
from ase.build import bulk, molecule
from ase.io import write
def test_exec_atoms(cli, fname, atoms):
    out = cli.ase('exec', fname, '-e', 'print(atoms.symbols)')
    assert out.strip() == str(atoms.symbols)