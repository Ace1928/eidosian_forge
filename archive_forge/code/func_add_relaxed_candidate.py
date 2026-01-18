import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def add_relaxed_candidate(self, candidate, **kwargs):
    """ Add a relaxed starting candidate. """
    test_raw_score(candidate)
    if 'data' in candidate.info:
        data = candidate.info['data']
    else:
        data = {}
    gaid = self.c.write(candidate, origin='StartingCandidateRelaxed', relaxed=1, generation=0, extinct=0, key_value_pairs=candidate.info['key_value_pairs'], data=data, **kwargs)
    self.c.update(gaid, gaid=gaid)
    candidate.info['confid'] = gaid