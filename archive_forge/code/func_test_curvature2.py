import pytest
from ase import Atoms
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
def test_curvature2(testdir):
    """This test has two atoms spinning counter-clockwise around eachother. the
    The radius (1/curvature) is less obviously pair_distance*sqrt(2)/2.
    This is the simplest multi-body analytic curvature test."""
    name = 'test_curvature2'
    radius = pair_distance * np.sqrt(2) / 2
    atoms = Al_atom_pair(pair_distance)
    atoms.set_velocities([[0, -1, 0], [0, 1, 0]])
    with ContourExploration(atoms, maxstep=1.0, parallel_drift=0.0, angle_limit=30, trajectory=name + '.traj', logfile=name + '.log') as dyn:
        print('Target Radius (1/curvature) {: .6f} Ang'.format(radius))
        for i in range(5):
            dyn.run(30)
            print('Radius (1/curvature) {: .6f} Ang'.format(1 / dyn.curvature))
            assert radius == pytest.approx(1.0 / dyn.curvature, abs=0.002)