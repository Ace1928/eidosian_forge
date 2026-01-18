import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def read_adp_data(self, data, d):
    """read in the extra adp data from the potential file"""
    self.d_data = np.zeros([self.Nelements, self.Nelements, self.nr])
    for i in range(self.Nelements):
        for j in range(i + 1):
            self.d_data[j, i] = data[d:d + self.nr]
            d += self.nr
    self.q_data = np.zeros([self.Nelements, self.Nelements, self.nr])
    for i in range(self.Nelements):
        for j in range(i + 1):
            self.q_data[j, i] = data[d:d + self.nr]
            d += self.nr