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
def read_k_point_weights(self, filename):
    """Read k-point weighting. Normally named IBZKPT."""
    lines = self.load_file(filename)
    if 'Tetrahedra\n' in lines:
        N = lines.index('Tetrahedra\n')
    else:
        N = len(lines)
    kpt_weights = []
    for n in range(3, N):
        kpt_weights.append(float(lines[n].split()[3]))
    kpt_weights = np.array(kpt_weights)
    kpt_weights /= np.sum(kpt_weights)
    return kpt_weights