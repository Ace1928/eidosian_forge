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
def make_dimer(constraint=True):
    """Prepare atoms object for testing"""
    dimer = s22.create_s22_system('Water_dimer')
    calc = TIP3P(rc=9.0)
    dimer.calc = calc
    if constraint:
        dimer.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3) for i in range(2) for j in [0, 1, 2]])
    return dimer