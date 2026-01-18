import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
def create_solver(self):
    if self.use_pyamg and have_pyamg:
        start_time = time.time()
        self.ml = create_pyamg_solver(self.P)
        self.logfile.write('--- multi grid solver created in %s ---\n' % (time.time() - start_time))