import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def next_entry():
    line = lines.pop(0).strip()
    if len(line) > 0:
        lines.insert(0, line)