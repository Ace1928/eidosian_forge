from ase.ga.offspring_creator import OffspringCreator
from ase import Atoms
from itertools import chain
import numpy as np
def get_numbers(self, atoms):
    """Returns the atomic numbers of the atoms object using only
        the elements defined in self.elements"""
    ac = atoms.copy()
    if self.elements is not None:
        del ac[[a.index for a in ac if a.symbol in self.elements]]
    return ac.numbers