import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def sendposdata(self, cell, icell, positions):
    assert cell.size == 9
    assert icell.size == 9
    assert positions.size % 3 == 0
    self.log(' sendposdata')
    self.sendmsg('POSDATA')
    self.send(cell.T / units.Bohr, np.float64)
    self.send(icell.T * units.Bohr, np.float64)
    self.send(len(positions), np.int32)
    self.send(positions / units.Bohr, np.float64)