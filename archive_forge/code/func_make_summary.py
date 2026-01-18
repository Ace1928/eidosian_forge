import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def make_summary(self, atoms1, atoms2):
    return '\n'.join([summary_function(atoms1, atoms2) for summary_function in self.summary_functions])