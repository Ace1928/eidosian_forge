import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def mark_as_queued(self, a):
    """ Marks a configuration as queued for relaxation. """
    gaid = a.info['confid']
    self.c.write(None, gaid=gaid, queued=1, key_value_pairs=a.info['key_value_pairs'])