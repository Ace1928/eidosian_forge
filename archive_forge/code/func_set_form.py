import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def set_form(self, name):
    """set the form variable based on the file name suffix"""
    extension = os.path.splitext(name)[1]
    if extension == '.eam':
        self.form = 'eam'
    elif extension == '.alloy':
        self.form = 'alloy'
    elif extension == '.adp':
        self.form = 'adp'
    elif extension == '.fs':
        self.form = 'fs'
    else:
        raise RuntimeError('unknown file extension type: %s' % extension)