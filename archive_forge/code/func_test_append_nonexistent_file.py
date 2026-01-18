import pytest
from ase import Atom, Atoms
from ase.io import Trajectory, read
from ase.constraints import FixBondLength
from ase.calculators.calculator import PropertyNotImplementedError
def test_append_nonexistent_file(co):
    fname = '2.traj'
    with Trajectory(fname, 'a', co) as t:
        pass
    with Trajectory('empty.traj', 'w') as t:
        pass
    with Trajectory('empty.traj', 'r') as t:
        assert len(t) == 0