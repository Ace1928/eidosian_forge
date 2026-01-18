import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def kill_candidate(self, confid):
    """Sets extinct=1 in the key_value_pairs of the candidate
        with gaid=confid. This could be used in the
        mass extinction operator."""
    for dct in self.c.select(gaid=confid):
        self.c.update(dct.id, extinct=1)