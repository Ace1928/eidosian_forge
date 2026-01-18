import numpy as np
from ase.io.fortranfile import FortranFile
def get_valence_charge(filename):
    """ Read the valence charge from '.psf'-file."""
    with open(filename, 'r') as fd:
        fd.readline()
        fd.readline()
        fd.readline()
        valence = -float(fd.readline().split()[-1])
    return valence