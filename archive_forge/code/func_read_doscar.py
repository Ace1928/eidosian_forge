import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def read_doscar(self, fname='DOSCAR'):
    """Read a VASP DOSCAR file"""
    fd = open(fname)
    natoms = int(fd.readline().split()[0])
    [fd.readline() for nn in range(4)]
    ndos = int(fd.readline().split()[2])
    dos = []
    for nd in range(ndos):
        dos.append(np.array([float(x) for x in fd.readline().split()]))
    self._total_dos = np.array(dos).T
    dos = []
    for na in range(natoms):
        line = fd.readline()
        if line == '':
            break
        ndos = int(line.split()[2])
        line = fd.readline().split()
        cdos = np.empty((ndos, len(line)))
        cdos[0] = np.array(line)
        for nd in range(1, ndos):
            line = fd.readline().split()
            cdos[nd] = np.array([float(x) for x in line])
        dos.append(cdos.T)
    self._site_dos = np.array(dos)
    fd.close()