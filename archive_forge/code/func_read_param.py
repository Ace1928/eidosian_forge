import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def read_param(filename='', calc=None, fd=None, get_interface_options=False):
    if fd is None:
        if filename == '':
            raise ValueError('One between filename and fd must be provided')
        fd = open(filename)
    elif filename:
        warnings.warn('Filestream used to read param, file name will be ignored')
    if get_interface_options:
        int_opts = {}
        optre = re.compile('# ASE_INTERFACE ([^\\s]+) : ([^\\s]+)')
        lines = fd.readlines()
        fd.seek(0)
        for l in lines:
            m = optre.search(l)
            if m:
                int_opts[m.groups()[0]] = m.groups()[1]
    data = read_freeform(fd)
    if calc is None:
        from ase.calculators.castep import Castep
        calc = Castep(check_castep_version=False, keyword_tolerance=2)
    for kw, (val, otype) in data.items():
        if otype == 'block':
            val = val.split('\n')
        calc.param.__setattr__(kw, val)
    if not get_interface_options:
        return calc
    else:
        return (calc, int_opts)