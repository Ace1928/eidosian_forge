import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def add_more_relaxed_steps(self, a_list):
    print('Please use add_more_relaxed_candidates instead')
    self.add_more_relaxed_candidates(a_list)