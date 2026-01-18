import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def set_adp_splines(self):
    self.d = np.empty([self.Nelements, self.Nelements], object)
    self.d_d = np.empty([self.Nelements, self.Nelements], object)
    self.q = np.empty([self.Nelements, self.Nelements], object)
    self.d_q = np.empty([self.Nelements, self.Nelements], object)
    for i in range(self.Nelements):
        for j in range(i, self.Nelements):
            self.d[i, j] = spline(self.r[1:], self.d_data[i, j][1:], k=3)
            self.d_d[i, j] = self.deriv(self.d[i, j])
            self.q[i, j] = spline(self.r[1:], self.q_data[i, j][1:], k=3)
            self.d_q[i, j] = self.deriv(self.q[i, j])
            if j != i:
                self.d[j, i] = self.d[i, j]
                self.d_d[j, i] = self.d_d[i, j]
                self.q[j, i] = self.q[i, j]
                self.d_q[j, i] = self.d_q[i, j]