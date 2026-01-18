import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def is_duplicate(self, **kwargs):
    """Check if the key-value pair is already present in the database"""
    return len(list(self.c.select(**kwargs))) > 0