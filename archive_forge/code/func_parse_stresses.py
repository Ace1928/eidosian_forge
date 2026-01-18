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
def parse_stresses(self):
    """Get the stresses from stress file, or from the main output file."""
    if len(self.filenames['stress']) == 1:
        dat = np.genfromtxt(self.filenames['stress'][0], skip_header=1)
        dat = [dat] if len(np.shape(dat)) == 1 else dat
        self.data['stress_tensor'] = [[list(d[2:5]), list(d[5:8]), list(d[8:11])] for d in dat]
    else:
        header_pattern = 'STRESS\\|\\s+x\\s+y\\s+z'
        row_pattern = 'STRESS\\|\\s+[?:x|y|z]\\s+(-?\\d+\\.\\d+E?[-|\\+]?\\d+)\\s+(-?\\d+\\.\\d+E?[-|\\+]?\\d+)\\s+(-?\\d+\\.\\d+E?[-|\\+]?\\d+).*$'
        footer_pattern = '^$'
        d = self.read_table_pattern(header_pattern=header_pattern, row_pattern=row_pattern, footer_pattern=footer_pattern, postprocess=postprocessor, last_one_only=False)

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                if i % 2 == 0:
                    yield lst[i:i + n]
        if d:
            self.data['stress_tensor'] = list(chunks(d[0], 3))