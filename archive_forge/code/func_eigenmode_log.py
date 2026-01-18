import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def eigenmode_log(self):
    """Log the eigenmodes (eigenmode estimates)"""
    if self.mlogfile is not None:
        l = 'MINMODE:MODE: Optimization Step: %i\n' % self.control.get_counter('optcount')
        for m_num, mode in enumerate(self.eigenmodes):
            l += 'MINMODE:MODE: Order: %i\n' % m_num
            for k in range(len(mode)):
                l += 'MINMODE:MODE: %7i %15.8f %15.8f %15.8f\n' % (k, mode[k][0], mode[k][1], mode[k][2])
        self.mlogfile.write(l)
        self.mlogfile.flush()