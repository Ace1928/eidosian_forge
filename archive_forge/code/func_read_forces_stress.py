import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def read_forces_stress(self):
    """Read the forces and stress from the FORCE_STRESS file.
        """
    fname = self.getpath('FORCE_STRESS')
    with open(fname, 'r') as fd:
        lines = fd.readlines()
    stress_lines = lines[1:4]
    stress = np.empty((3, 3))
    for i in range(3):
        line = stress_lines[i].strip().split(' ')
        line = [s for s in line if len(s) > 0]
        stress[i] = [float(s) for s in line]
    self.results['stress'] = np.array([stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1]])
    self.results['stress'] *= Ry / Bohr ** 3
    start = 5
    self.results['forces'] = np.zeros((len(lines) - start, 3), float)
    for i in range(start, len(lines)):
        line = [s for s in lines[i].strip().split(' ') if len(s) > 0]
        self.results['forces'][i - start] = [float(s) for s in line[2:5]]
    self.results['forces'] *= Ry / Bohr