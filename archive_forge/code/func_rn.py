import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def rn(line='\n', n=1):
    """
    Read n'th to last value.
    For example:
        ...
        scf.XcType          LDA
        scf.Kgrid         4 4 4
        ...
    In Python,
        >>> str(rn(line, 1))
        LDA
        >>> line = f.readline()
        >>> int(rn(line, 3))
        4
    """
    return line.split()[-n]