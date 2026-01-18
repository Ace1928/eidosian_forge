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
def parse_global_params(self):
    """Parse the GLOBAL section parameters from CP2K output file into a dictionary."""
    pat = re.compile('\\s+GLOBAL\\|\\s+([\\w+\\s]*)\\s+(\\w+)')
    self.read_pattern({'global': pat}, terminate_on_match=False, reverse=False)
    for d in self.data['global']:
        d[0], d[1] = (postprocessor(d[0]), str(d[1]))
    self.data['global'] = dict(self.data['global'])