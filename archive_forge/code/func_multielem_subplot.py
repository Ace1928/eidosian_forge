import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def multielem_subplot(self, curvex, curvey, xlabel, ylabel, name, plt, half=True):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in np.arange(self.Nelements):
        for j in np.arange(i + 1 if half else self.Nelements):
            label = name + ' ' + self.elements[i] + '-' + self.elements[j]
            plt.plot(curvex, curvey[i, j](curvex), label=label)
    plt.legend()