import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def input_line_iter(lines):
    """Convenient iterator for parsing input files 'cleanly'.

    Discards comments etc."""
    for line in lines:
        line = line.split('#')[0].strip()
        if not line or line.isspace():
            continue
        line = line.strip()
        yield line