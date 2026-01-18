import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def get_default_abinit_pp_paths():
    return os.environ.get('ABINIT_PP_PATH', '.').split(':')