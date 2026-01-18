from ase.io import Trajectory
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.calculators.singlepoint import SinglePointCalculator
import ase.parallel as mpi
import numpy as np
import shutil
import os
import types
from math import log
from math import exp
from contextlib import ExitStack
def get_highest_energy_index(self):
    """Find the index of the image with the highest energy."""
    energies = self.get_energies()
    valid_entries = [(i, e) for i, e in enumerate(energies) if e == e]
    highest_energy_index = max(valid_entries, key=lambda x: x[1])[0]
    return highest_energy_index