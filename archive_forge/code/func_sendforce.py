import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def sendforce(self, energy, forces, virial, morebytes=np.zeros(1, dtype=np.byte)):
    assert np.array([energy]).size == 1
    assert forces.shape[1] == 3
    assert virial.shape == (3, 3)
    self.log(' sendforce')
    self.sendmsg('FORCEREADY')
    self.send(np.array([energy / units.Ha]), np.float64)
    natoms = len(forces)
    self.send(np.array([natoms]), np.int32)
    self.send(units.Bohr / units.Ha * forces, np.float64)
    self.send(1.0 / units.Ha * virial.T, np.float64)
    self.send(np.array([len(morebytes)]), np.int32)
    self.send(morebytes, np.byte)