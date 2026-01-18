import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_bool(line):
    bool = str(rn(line)).lower()
    if bool == 'on':
        return True
    elif bool == 'off':
        return False
    else:
        return None