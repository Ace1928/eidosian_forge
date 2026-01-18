import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def summary_functions_on_conditions(has_calc):
    if has_calc:
        return [rmsd, energy_delta]
    return [rmsd]