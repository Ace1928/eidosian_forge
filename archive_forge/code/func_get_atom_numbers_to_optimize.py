import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_atom_numbers_to_optimize(self):
    """ Get the list of atom numbers being optimized. """
    v = self.c.get(simulation_cell=True)
    return v.data.stoichiometry