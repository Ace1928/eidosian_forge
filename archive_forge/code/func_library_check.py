import os
import re
import numpy as np
from ase.units import eV, Ang
from ase.calculators.calculator import FileIOCalculator, ReadError
def library_check(self):
    if self.parameters['library'] is not None:
        if 'GULP_LIB' not in os.environ:
            raise RuntimeError('Be sure to have set correctly $GULP_LIB or to have the force field library.')