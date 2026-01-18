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
def ran_successfully(self):
    """
        Sanity checks that the program ran successfully. Looks at the bottom of the CP2K output
        file for the "PROGRAM ENDED" line, which is printed when successfully ran. Also grabs
        the number of warnings issued.
        """
    program_ended_at = re.compile('PROGRAM ENDED AT\\s+(\\w+)')
    num_warnings = re.compile('The number of warnings for this run is : (\\d+)')
    self.read_pattern(patterns={'completed': program_ended_at}, reverse=True, terminate_on_match=True, postprocess=bool)
    self.read_pattern(patterns={'num_warnings': num_warnings}, reverse=True, terminate_on_match=True, postprocess=int)
    if not self.completed:
        raise ValueError('The provided CP2K job did not finish running! Cannot parse the file reliably.')