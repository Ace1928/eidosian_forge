import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def boxshape_is_ase_compatible(kwargs):
    pdims = int(kwargs.get('periodicdimensions', 0))
    default_boxshape = 'parallelepiped' if pdims > 0 else 'minimum'
    boxshape = kwargs.get('boxshape', default_boxshape).lower()
    return boxshape == 'parallelepiped'