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
def parse_cell_params(self):
    """Parse the lattice parameters (initial) from the output file."""
    if self.input and self.input.check('force_eval/subsys/cell'):
        cell = self.input['force_eval']['subsys']['cell']
        if cell.get('abc'):
            return [[cell['abc'].values[0], 0, 0], [0, cell['abc'].values[1], 0], [0, 0, cell['abc'].values[2]]]
        return [list(cell.get('A').values), list(cell.get('B').values), list(cell.get('C').values)]
    warnings.warn('Input file lost. Reading cell params from summary at top of output. Precision errors may result.')
    cell_volume = re.compile('\\s+CELL\\|\\sVolume.*\\s(\\d+\\.\\d+)')
    vectors = re.compile('\\s+CELL\\| Vector.*\\s(-?\\d+\\.\\d+)\\s+(-?\\d+\\.\\d+)\\s+(-?\\d+\\.\\d+)')
    angles = re.compile('\\s+CELL\\| Angle.*\\s(\\d+\\.\\d+)')
    self.read_pattern({'cell_volume': cell_volume, 'lattice': vectors, 'angles': angles}, terminate_on_match=False, postprocess=float, reverse=False)
    i = iter(self.data['lattice'])
    lattices = list(zip(i, i, i))
    return lattices[0]