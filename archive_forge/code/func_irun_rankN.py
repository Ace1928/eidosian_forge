import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def irun_rankN(self, atoms, use_stress=True):
    stop_criterion = np.zeros(1, bool)
    while True:
        self.comm.broadcast(stop_criterion, 0)
        if stop_criterion[0]:
            return
        self.calculate(atoms, use_stress)
        yield