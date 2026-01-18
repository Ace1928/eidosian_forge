import os
import operator as op
import re
import warnings
from collections import OrderedDict
from os import path
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.dft.kpoints import kpoint_convert
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols, atomic_numbers
from ase.units import create_units
from ase.utils import iofunction
def middle_brackets(full_text):
    """Extract text from innermost brackets."""
    start, end = (0, len(full_text))
    for idx, char in enumerate(full_text):
        if char == '(':
            start = idx
        if char == ')':
            end = idx + 1
            break
    return full_text[start:end]