import gzip
import struct
from collections import deque
from os.path import splitext
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.quaternions import Quaternions
def get_quantity(labels, quantity=None):
    try:
        cols = [colnames.index(label) for label in labels]
        if quantity:
            return convert(data[:, cols].astype(float), quantity, units, 'ASE')
        return data[:, cols].astype(float)
    except ValueError:
        return None