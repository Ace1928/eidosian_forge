import pytest
from ase.build import bulk
from ase.io import write
def test_info_formats(cli):
    assert 'traj' in cli.ase('info', '--formats')