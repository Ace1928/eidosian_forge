import pytest
from ase.build import bulk, molecule
from ase.io import write
def test_exec_index(cli, fname):
    out = cli.ase('exec', fname, '-e', 'print(index)')
    assert out.strip() == str(0)