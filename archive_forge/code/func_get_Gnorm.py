import os
import re
import numpy as np
from ase.units import eV, Ang
from ase.calculators.calculator import FileIOCalculator, ReadError
def get_Gnorm(self):
    return self.Gnorm