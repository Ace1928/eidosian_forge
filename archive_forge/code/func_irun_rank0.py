import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def irun_rank0(self, atoms, use_stress=True):
    try:
        while True:
            try:
                msg = self.protocol.recvmsg()
            except SocketClosed:
                msg = 'EXIT'
            if msg == 'EXIT':
                self.comm.broadcast(np.ones(1, bool), 0)
                return
            elif msg == 'STATUS':
                self.protocol.sendmsg(self.state)
            elif msg == 'POSDATA':
                assert self.state == 'READY'
                cell, icell, positions = self.protocol.recvposdata()
                atoms.cell[:] = cell
                atoms.positions[:] = positions
                self.comm.broadcast(np.zeros(1, bool), 0)
                energy, forces, virial = self.calculate(atoms, use_stress)
                self.state = 'HAVEDATA'
                yield
            elif msg == 'GETFORCE':
                assert self.state == 'HAVEDATA', self.state
                self.protocol.sendforce(energy, forces, virial)
                self.state = 'NEEDINIT'
            elif msg == 'INIT':
                assert self.state == 'NEEDINIT'
                bead_index, initbytes = self.protocol.recvinit()
                self.bead_index = bead_index
                self.bead_initbytes = initbytes
                self.state = 'READY'
            else:
                raise KeyError('Bad message', msg)
    finally:
        self.close()