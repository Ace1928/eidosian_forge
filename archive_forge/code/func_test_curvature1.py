import pytest
from ase import Atoms
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
def test_curvature1(testdir):
    """This basic test has an atom spinning counter-clockwise around a fixed
    atom. The radius (1/curvature) must therefore be very
    close the pair_distance."""
    name = 'test_curvature1'
    radius = pair_distance
    atoms = Al_atom_pair(pair_distance)
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_velocities([[0, 0, 0], [0, 1, 0]])
    with ContourExploration(atoms, maxstep=1.5, parallel_drift=0.0, angle_limit=30, trajectory=name + '.traj', logfile=name + '.log') as dyn:
        print('Target Radius (1/curvature) {: .6f} Ang'.format(radius))
        for i in range(5):
            dyn.run(30)
            print('Radius (1/curvature) {: .6f} Ang'.format(1 / dyn.curvature))
            assert radius == pytest.approx(1.0 / dyn.curvature, abs=0.002)