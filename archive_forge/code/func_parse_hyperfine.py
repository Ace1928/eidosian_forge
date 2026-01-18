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
def parse_hyperfine(self, hyperfine_filename=None):
    """Parse a file containing hyperfine coupling tensors for each atomic site."""
    if not hyperfine_filename:
        if self.filenames['hyperfine_tensor']:
            hyperfine_filename = self.filenames['hyperfine_tensor'][0]
        else:
            return None
    with zopen(hyperfine_filename, mode='rt') as file:
        lines = [line for line in file.read().split('\n') if line]
    hyperfine = [[] for _ in self.ionic_steps]
    for i in range(2, len(lines), 5):
        x = list(map(float, lines[i + 2].split()[-3:]))
        y = list(map(float, lines[i + 3].split()[-3:]))
        z = list(map(float, lines[i + 4].split()[-3:]))
        hyperfine[-1].append([x, y, z])
    self.data['hyperfine_tensor'] = hyperfine
    return hyperfine