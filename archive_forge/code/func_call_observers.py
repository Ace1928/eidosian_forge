import time
from math import sqrt
from os.path import isfile
from ase.io.jsonio import read_json, write_json
from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world, barrier
from ase.io.trajectory import Trajectory
from ase.utils import IOContext
import collections.abc
def call_observers(self):
    for function, interval, args, kwargs in self.observers:
        call = False
        if interval > 0:
            if self.nsteps % interval == 0:
                call = True
        elif interval <= 0:
            if self.nsteps == abs(interval):
                call = True
        if call:
            function(*args, **kwargs)