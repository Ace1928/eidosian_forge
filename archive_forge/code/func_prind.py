import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def prind(*line, end='\n'):
    if debug:
        print(*line, end=end)