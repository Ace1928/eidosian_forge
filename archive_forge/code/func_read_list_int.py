import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_list_int(line):
    return [int(x) for x in line.split()[1:]]