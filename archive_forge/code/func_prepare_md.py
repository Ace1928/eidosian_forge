import numpy as np
from ase.build import bulk
from ase.units import fs
from ase.md import VelocityVerlet
from ase.md import Langevin
from ase.md import Andersen
from ase.io import Trajectory, read
import pytest
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
def prepare_md(atoms, calculator):
    a_md = atoms.copy()
    a_md.calc = calculator
    traj = Trajectory('Au7Ag.traj', 'w', a_md)
    return (a_md, traj)