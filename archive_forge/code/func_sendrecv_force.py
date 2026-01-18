import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def sendrecv_force(self):
    self.log(' sendrecv_force')
    self.sendmsg('GETFORCE')
    msg = self.recvmsg()
    assert msg == 'FORCEREADY', msg
    e = self.recv(1, np.float64)[0]
    natoms = self.recv(1, np.int32)
    assert natoms >= 0
    forces = self.recv((int(natoms), 3), np.float64)
    virial = self.recv((3, 3), np.float64).T.copy()
    nmorebytes = self.recv(1, np.int32)
    nmorebytes = int(nmorebytes)
    if nmorebytes > 0:
        morebytes = self.recv(nmorebytes, np.byte)
    else:
        morebytes = b''
    return (e * units.Ha, units.Ha / units.Bohr * forces, units.Ha * virial, morebytes)