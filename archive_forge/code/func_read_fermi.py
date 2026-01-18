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
def read_fermi(self, lines=None):
    """Method that reads Fermi energy from OUTCAR file"""
    if not lines:
        lines = self.load_file('OUTCAR')
    E_f = None
    for line in lines:
        if 'E-fermi' in line:
            E_f = float(line.split()[2])
    return E_f