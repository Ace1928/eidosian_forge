import pytest
from ase import Atoms
from ase.build import bulk, fcc111
from ase.io import write
def test_bzplot(cli, file, plt):
    cli.ase('reciprocal', file, 'bandpath.svg')