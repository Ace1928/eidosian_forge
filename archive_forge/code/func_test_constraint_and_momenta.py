import pytest
from ase import Atom, Atoms
from ase.io import Trajectory, read
from ase.constraints import FixBondLength
from ase.calculators.calculator import PropertyNotImplementedError
def test_constraint_and_momenta():
    a = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)], momenta=[(1, 0, 0), (0, 0, 0)])
    a.constraints = [FixBondLength(0, 1)]
    with Trajectory('constraint.traj', 'w', a) as t:
        t.write()
    b = read('constraint.traj')
    assert not (b.get_momenta() - a.get_momenta()).any()