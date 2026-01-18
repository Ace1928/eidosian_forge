from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
def parse_initial_structure(self):
    """Parse the initial structure from the main cp2k output file."""
    pattern = re.compile('- Atoms:\\s+(\\d+)')
    patterns = {'num_atoms': pattern}
    self.read_pattern(patterns=patterns, reverse=False, terminate_on_match=True, postprocess=int)
    coord_table = []
    with zopen(self.filename, mode='rt') as file:
        while True:
            line = file.readline()
            if 'Atom  Kind  Element       X           Y           Z          Z(eff)       Mass' in line:
                for _ in range(self.data['num_atoms'][0][0]):
                    line = file.readline().split()
                    if line == []:
                        line = file.readline().split()
                    coord_table.append(line)
                break
    lattice = self.parse_cell_params()
    gs = {}
    self.data['atomic_kind_list'] = []
    for v in self.data['atomic_kind_info'].values():
        if v['pseudo_potential'].upper() == 'NONE':
            gs[v['kind_number']] = True
        else:
            gs[v['kind_number']] = False
    for c in coord_table:
        for k, v in self.data['atomic_kind_info'].items():
            if int(v['kind_number']) == int(c[1]):
                v['element'] = c[2]
                self.data['atomic_kind_list'].append(k)
                break
    if self.is_molecule:
        self.initial_structure = Molecule(species=[i[2] for i in coord_table], coords=[[float(i[4]), float(i[5]), float(i[6])] for i in coord_table], site_properties={'ghost': [gs.get(int(i[1])) for i in coord_table]}, charge=self.charge, spin_multiplicity=self.multiplicity)
    else:
        self.initial_structure = Structure(lattice, species=[i[2] for i in coord_table], coords=[[float(i[4]), float(i[5]), float(i[6])] for i in coord_table], coords_are_cartesian=True, site_properties={'ghost': [gs.get(int(i[1])) for i in coord_table]}, charge=self.charge)
    self.composition = self.initial_structure.composition
    return self.initial_structure