import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def read_mag(self, lines=None):
    if not lines:
        lines = self.load_file('OUTCAR')
    p = self.int_params
    q = self.list_float_params
    if self.spinpol:
        magnetic_moment = self._read_magnetic_moment(lines=lines)
        if p['lorbit'] is not None and p['lorbit'] >= 10 or (p['lorbit'] is None and q['rwigs']):
            magnetic_moments = self._read_magnetic_moments(lines=lines)
        else:
            warn('Magnetic moment data not written in OUTCAR (LORBIT<10), setting magnetic_moments to zero.\nSet LORBIT>=10 to get information on magnetic moments')
            magnetic_moments = np.zeros(len(self.atoms))
    else:
        magnetic_moment = 0.0
        magnetic_moments = np.zeros(len(self.atoms))
    return (magnetic_moment, magnetic_moments)