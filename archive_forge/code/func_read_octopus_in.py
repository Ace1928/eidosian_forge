import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
@reader
def read_octopus_in(fd, get_kwargs=False):
    kwargs = parse_input_file(fd)
    try:
        fname = fd.name
    except AttributeError:
        directory = None
    else:
        directory = os.path.split(fname)[0]
    atoms, remaining_kwargs = kwargs2atoms(kwargs, directory=directory)
    if get_kwargs:
        return (atoms, remaining_kwargs)
    else:
        return atoms