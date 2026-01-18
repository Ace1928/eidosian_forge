import pytest
from ase.build import bulk, molecule
from ase.io import write
def test_exec_images(cli, fname, atoms):
    out = cli.ase('exec', fname, '-e', 'print(len(images[0]))')
    assert out.strip() == str(len(atoms))