import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def floa(byte, shape=None):
    return easyReader(byte, 'd', shape)