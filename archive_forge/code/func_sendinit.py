import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def sendinit(self):
    self.log(' sendinit')
    self.sendmsg('INIT')
    self.send(0, np.int32)
    self.send(1, np.int32)
    self.send(np.zeros(1), np.byte)