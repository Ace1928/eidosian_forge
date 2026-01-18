import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def slice_split(filename):
    if '@' in filename:
        filename, index = parse_filename(filename, None)
    else:
        filename, index = parse_filename(filename, default_index)
    return (filename, index)