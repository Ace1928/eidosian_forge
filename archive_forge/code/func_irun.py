import time
from math import sqrt
from os.path import isfile
from ase.io.jsonio import read_json, write_json
from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world, barrier
from ase.io.trajectory import Trajectory
from ase.utils import IOContext
import collections.abc
def irun(self, fmax=0.05, steps=None):
    """ call Dynamics.irun and keep track of fmax"""
    self.fmax = fmax
    if steps:
        self.max_steps = steps
    return Dynamics.irun(self)