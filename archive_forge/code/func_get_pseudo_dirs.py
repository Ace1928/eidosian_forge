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
def get_pseudo_dirs(data):
    """Guess a list of possible locations for pseudopotential files.

    Parameters
    ----------
    data : Namelist
        Namelist representing the quantum espresso input parameters

    Returns
    -------
    pseudo_dirs : list[str]
        A list of directories where pseudopotential files could be located.
    """
    pseudo_dirs = []
    if 'pseudo_dir' in data['control']:
        pseudo_dirs.append(data['control']['pseudo_dir'])
    if 'ESPRESSO_PSEUDO' in os.environ:
        pseudo_dirs.append(os.environ['ESPRESSO_PSEUDO'])
    pseudo_dirs.append(path.expanduser('~/espresso/pseudo/'))
    return pseudo_dirs