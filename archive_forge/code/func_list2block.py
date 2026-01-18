import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def list2block(name, rows):
    """Construct 'block' of Octopus input.

    convert a list of rows to a string with the format x | x | ....
    for the octopus input file"""
    lines = []
    lines.append('%' + name)
    for row in rows:
        lines.append(' ' + ' | '.join((str(obj) for obj in row)))
    lines.append('%')
    return lines