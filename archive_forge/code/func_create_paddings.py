from collections import defaultdict
import numpy as np
import kimpy
from kimpy import neighlist
from ase.neighborlist import neighbor_list
from ase import Atom
from .kimpy_wrappers import check_call_wrapper
@check_call_wrapper
def create_paddings(self, cell, pbc, contributing_coords, contributing_species_code):
    cell = np.asarray(cell, dtype=np.double)
    pbc = np.asarray(pbc, dtype=np.intc)
    contributing_coords = np.asarray(contributing_coords, dtype=np.double)
    return neighlist.create_paddings(self.influence_dist, cell, pbc, contributing_coords, contributing_species_code)