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
def parse_timing(self):
    """Parse the timing info (how long did the run take)."""
    header = 'SUBROUTINE\\s+CALLS\\s+ASD\\s+SELF TIME\\s+TOTAL TIME\\s+MAXIMUM\\s+AVERAGE\\s+MAXIMUM\\s+AVERAGE\\s+MAXIMUM'
    row = '(\\w+)\\s+(.+)\\s+(\\d+\\.\\d+)\\s+(\\d+\\.\\d+)\\s+(\\d+\\.\\d+)\\s+(\\d+\\.\\d+)\\s+(\\d+\\.\\d+)'
    footer = '\\-+'
    timing = self.read_table_pattern(header_pattern=header, row_pattern=row, footer_pattern=footer, last_one_only=True, postprocess=postprocessor)
    self.timing = {}
    for time in timing:
        self.timing[time[0]] = {'calls': {'max': time[1]}, 'asd': time[2], 'self_time': {'average': time[3], 'maximum': time[4]}, 'total_time': {'average': time[5], 'maximum': time[6]}}