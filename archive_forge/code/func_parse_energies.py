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
def parse_energies(self):
    """
        Get the total energy from a CP2K calculation. Presently, the energy reported in the
        trajectory (pos.xyz) file takes precedence over the energy reported in the main output
        file. This is because the trajectory file keeps track of energies in between restarts,
        while the main output file may or may not depending on whether a particular machine
        overwrites or appends it.
        """
    if self.filenames.get('trajectory'):
        toten_pattern = '.*E\\s+\\=\\s+(-?\\d+.\\d+)'
        matches = regrep(self.filenames['trajectory'][-1], {'total_energy': toten_pattern}, postprocess=float)
        self.data['total_energy'] = list(chain.from_iterable(np.multiply([i[0] for i in matches.get('total_energy', [[]])], Ha_to_eV)))
    else:
        toten_pattern = re.compile('Total FORCE_EVAL.*\\s(-?\\d+.\\d+)')
        self.read_pattern({'total_energy': toten_pattern}, terminate_on_match=False, postprocess=float, reverse=False)
        self.data['total_energy'] = list(chain.from_iterable(np.multiply(self.data.get('total_energy', []), Ha_to_eV)))
    self.final_energy = self.data.get('total_energy', [])[-1]