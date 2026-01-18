import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_chemical_potential(line, f, debug=None):
    return read_float(line)