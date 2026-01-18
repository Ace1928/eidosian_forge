import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def read_potcar_numbers_of_electrons(file_obj):
    """ Read list of tuples (atomic symbol, number of valence electrons)
    for each atomtype from a POTCAR file."""
    nelect = []
    lines = file_obj.readlines()
    for n, line in enumerate(lines):
        if 'TITEL' in line:
            symbol = line.split('=')[1].split()[1].split('_')[0].strip()
            valence = float(lines[n + 4].split(';')[1].split('=')[1].split()[0].strip())
            nelect.append((symbol, valence))
    return nelect