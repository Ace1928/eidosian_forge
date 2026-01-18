import numpy as np
import pytest
from ase.optimize import FIRE, BFGS
from ase.data import s22
from ase.calculators.tip3p import TIP3P
from ase.constraints import FixBondLengths
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io import Trajectory
import ase.units as units
@pytest.mark.parametrize('loginterval', [1, 2])
@pytest.mark.parametrize('cls, kwargs', md_cls_and_kwargs)
def test_md_log_and_trajectory_length(cls, testdir, kwargs, loginterval):
    timestep = 1 * units.fs
    trajectory = 'md.traj'
    logfile = 'md.log'
    atoms = make_dimer(constraint=False)
    assert not atoms.constraints
    print('Testing', str(cls))
    with cls(atoms, logfile=logfile, timestep=timestep, trajectory=trajectory, loginterval=loginterval, **kwargs) as md:
        md.run(steps=5)
        md.run(steps=5)
    with open(logfile, 'rt') as fd:
        lines = list(fd)
    loglines = len(lines)
    print('Number of lines in log file:', loglines)
    with Trajectory(trajectory) as traj:
        trajframes = len(traj)
    print('Number of frames in trajectory:', trajframes)
    assert loglines == trajframes + 1