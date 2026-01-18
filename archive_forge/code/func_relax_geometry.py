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
def relax_geometry(self, atoms=None):
    """execute geometry optimization with script jobex"""
    if atoms is None:
        atoms = self.atoms
    self.set_atoms(atoms)
    if self.converged and (not self.update_geometry):
        return
    self.initialize()
    jobex_flags = ''
    if self.parameters['use resolution of identity']:
        jobex_flags += ' -ri'
    if self.parameters['force convergence']:
        par = self.parameters['force convergence']
        conv = floor(-log10(par / Ha * Bohr))
        jobex_flags += ' -gcart ' + str(int(conv))
    if self.parameters['energy convergence']:
        par = self.parameters['energy convergence']
        conv = floor(-log10(par / Ha))
        jobex_flags += ' -energy ' + str(int(conv))
    geom_iter = self.parameters['geometry optimization iterations']
    if geom_iter is not None:
        assert isinstance(geom_iter, int)
        jobex_flags += ' -c ' + str(geom_iter)
    self.converged = False
    execute('jobex' + jobex_flags)
    self.converged = self.read_convergence()
    if self.converged:
        self.update_energy = False
        self.update_forces = False
        self.update_geometry = False
        self.update_hessian = True
    new_struct = read('coord')
    atoms.set_positions(new_struct.get_positions())
    self.atoms = atoms.copy()
    self.read_energy()