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
def parse_ionic_steps(self):
    """Parse the ionic step info. If already parsed, this will just assimilate."""
    if not self.structures:
        self.parse_structures()
    if not self.data.get('total_energy'):
        self.parse_energies()
    if not self.data.get('forces'):
        self.parse_forces()
    if not self.data.get('stress_tensor'):
        self.parse_stresses()
    for i, (structure, energy) in enumerate(zip(self.structures, self.data.get('total_energy'))):
        self.ionic_steps.append({'structure': structure, 'E': energy, 'forces': self.data['forces'][i] if self.data.get('forces') else None, 'stress_tensor': self.data['stress_tensor'][i] if self.data.get('stress_tensor') else None})
    return self.ionic_steps