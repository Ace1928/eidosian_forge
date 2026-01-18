import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
def read_symops(self, castep_castep=None):
    """Read all symmetry operations used from a .castep file."""
    if castep_castep is None:
        castep_castep = self._seed + '.castep'
    if isinstance(castep_castep, str):
        if not os.path.isfile(castep_castep):
            warnings.warn('Warning: CASTEP file %s not found!' % castep_castep)
        f = paropen(castep_castep, 'r')
        _close = True
    else:
        f = castep_castep
        attributes = ['name', 'readline', 'close']
        for attr in attributes:
            if not hasattr(f, attr):
                raise TypeError('read_castep_castep_symops: castep_castep is not of type str nor valid fileobj!')
        castep_castep = f.name
        _close = False
    while True:
        line = f.readline()
        if not line:
            return
        if 'output verbosity' in line:
            iprint = line.split()[-1][1]
            if int(iprint) != 1:
                self.param.iprint = iprint
        if 'Symmetry and Constraints' in line:
            break
    if self.param.iprint.value is None or int(self.param.iprint.value) < 2:
        self._interface_warnings.append('Warning: No symmetryoperations could be read from %s (iprint < 2).' % f.name)
        return
    while True:
        line = f.readline()
        if not line:
            break
        if 'Number of symmetry operations' in line:
            nsym = int(line.split()[5])
            symmetry_operations = []
            for _ in range(nsym):
                rotation = []
                displacement = []
                while True:
                    if 'rotation' in f.readline():
                        break
                for _ in range(3):
                    line = f.readline()
                    rotation.append([float(r) for r in line.split()[1:4]])
                while True:
                    if 'displacement' in f.readline():
                        break
                line = f.readline()
                displacement = [float(d) for d in line.split()[1:4]]
                symop = {'rotation': rotation, 'displacement': displacement}
                self.symmetry_ops = symop
            self.symmetry = symmetry_operations
            warnings.warn('Symmetry operations successfully read from %s. %s' % (f.name, self.cell.symmetry_ops))
            break
    if _close:
        f.close()