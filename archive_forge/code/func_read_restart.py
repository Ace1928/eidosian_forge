import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def read_restart(self):
    """read a previous calculation from control file"""
    self.atoms = read('coord')
    self.atoms.calc = self
    self.converged = self.read_convergence()
    read_methods = [self.read_energy, self.read_gradient, self.read_forces, self.read_basis_set, self.read_ecps, self.read_mos, self.read_occupation_numbers, self.read_dipole_moment, self.read_ssquare, self.read_hessian, self.read_vibrational_reduced_masses, self.read_normal_modes, self.read_vibrational_spectrum, self.read_charges, self.read_point_charges, self.read_run_parameters]
    for method in read_methods:
        try:
            method()
        except ReadError as err:
            warnings.warn(err.args[0])