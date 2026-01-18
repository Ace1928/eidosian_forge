import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_number_of_unrelaxed_candidates(self):
    """ Returns the number of candidates not yet queued or relaxed. """
    return len(self.__get_ids_of_all_unrelaxed_candidates__())