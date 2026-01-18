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
def read_kpts(self, mode='ibz_k_points'):
    """ Returns list of kpts weights or kpts coordinates.  """
    values = []
    assert mode in ['ibz_k_points', 'k_point_weights']
    lines = open(self.out, 'r').readlines()
    kpts = None
    kptsstart = None
    for n, line in enumerate(lines):
        if line.rfind('| Number of k-points') > -1:
            kpts = int(line.split(':')[-1].strip())
    for n, line in enumerate(lines):
        if line.rfind('K-points in task') > -1:
            kptsstart = n
    assert kpts is not None
    assert kptsstart is not None
    text = lines[kptsstart + 1:]
    values = []
    for line in text[:kpts]:
        if mode == 'ibz_k_points':
            b = [float(c.strip()) for c in line.split()[4:7]]
        else:
            b = float(line.split()[-1])
        values.append(b)
    if len(values) == 0:
        values = None
    return np.array(values)