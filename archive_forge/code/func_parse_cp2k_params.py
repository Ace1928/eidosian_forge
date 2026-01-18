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
def parse_cp2k_params(self):
    """Parse the CP2K general parameters from CP2K output file into a dictionary."""
    version = re.compile('\\s+CP2K\\|.+version\\s+(.+)')
    input_file = re.compile('\\s+CP2K\\|\\s+Input file name\\s+(.+)$')
    self.read_pattern({'cp2k_version': version, 'input_filename': input_file}, terminate_on_match=True, reverse=False, postprocess=str)