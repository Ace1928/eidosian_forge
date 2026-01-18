import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_slab(self):
    """ Get the super cell, including stationary atoms, in which
            the structure is being optimized. """
    return self.c.get_atoms(simulation_cell=True)