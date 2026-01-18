import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def normalize_keywords(kwargs):
    """Reduce keywords to unambiguous form (lowercase)."""
    newkwargs = {}
    for arg, value in kwargs.items():
        lkey = arg.lower()
        newkwargs[lkey] = value
    return newkwargs