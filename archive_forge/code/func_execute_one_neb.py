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
def execute_one_neb(self, n_cur, to_run, climb=False, many_steps=False):
    with ExitStack() as exitstack:
        self._execute_one_neb(exitstack, n_cur, to_run, climb=climb, many_steps=many_steps)