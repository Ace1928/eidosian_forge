import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def add_unrelaxed_step(self, candidate, description):
    """ Add a change to a candidate without it having been relaxed.
            This method is typically used when a
            candidate has been mutated. """
    gaid = candidate.info['confid']
    t, desc = split_description(description)
    kwargs = {'relaxed': 0, 'extinct': 0, t: 1, 'description': desc, 'gaid': gaid}
    self.c.write(candidate, key_value_pairs=candidate.info['key_value_pairs'], data=candidate.info['data'], **kwargs)