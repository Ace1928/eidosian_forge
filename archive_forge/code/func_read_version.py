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
def read_version(self):
    """Get the VASP version number"""
    if not os.path.isfile(self._indir('OUTCAR')):
        return None
    with self.load_file_iter('OUTCAR') as lines:
        for line in lines:
            if ' vasp.' in line:
                return line[len(' vasp.'):].split()[0]
    return None