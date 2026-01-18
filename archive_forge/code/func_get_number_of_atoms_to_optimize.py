import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_number_of_atoms_to_optimize(self):
    """ Get the number of atoms being optimized. """
    v = self.c.get(simulation_cell=True)
    return len(v.data.stoichiometry)