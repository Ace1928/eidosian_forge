import os
import re
import numpy as np
from ase.units import eV, Ang
from ase.calculators.calculator import FileIOCalculator, ReadError
def get_atoms_labels(self):
    labels = np.array(self.atoms_labels)
    return labels