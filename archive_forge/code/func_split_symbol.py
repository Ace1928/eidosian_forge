import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def split_symbol(self, string, translate=default_map):
    if string in translate:
        return (translate[string], string)
    if len(string) < 2:
        return (string, None)
    return (string[0], string[1])