import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def read_stresses(self):
    """ Read stress per atom """
    with open(self.out) as fd:
        next((l for l in fd if 'Per atom stress (eV) used for heat flux calculation' in l))
        next((l for l in fd if '-------------' in l))
        stresses = []
        for l in [next(fd) for _ in range(len(self.atoms))]:
            xx, yy, zz, xy, xz, yz = [float(d) for d in l.split()[2:8]]
            stresses.append([xx, yy, zz, yz, xz, xy])
        self.results['stresses'] = np.array(stresses)