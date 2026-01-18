import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def read_extended_xyz(self, fileobj, map={}):
    """Read extended xyz file with labeled atoms."""
    atoms = read(fileobj)
    self.set_cell(atoms.get_cell())
    self.set_pbc(atoms.get_pbc())
    types = []
    types_map = {}
    for atom, type in zip(atoms, atoms.get_array('type')):
        if type not in types:
            types_map[type] = len(types)
            types.append(type)
        atom.tag = types_map[type]
        self.append(atom)
    self.types = types
    for name, array in atoms.arrays.items():
        if name not in self.arrays:
            self.new_array(name, array)