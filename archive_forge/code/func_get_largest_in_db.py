import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_largest_in_db(self, var):
    return next(self.c.select(sort='-{0}'.format(var))).get(var)