from math import sqrt
from sys import stdout
import numpy as np
import ase.units as units
from ase.parallel import parprint, paropen
from ase.vibrations import Vibrations
def intensity_prefactor(self, intensity_unit):
    if intensity_unit == '(D/A)2/amu':
        return (1.0, '(D/Å)^2 amu^-1')
    elif intensity_unit == 'km/mol':
        return (42.255, 'km/mol')
    else:
        raise RuntimeError('Intensity unit >' + intensity_unit + '< unknown.')