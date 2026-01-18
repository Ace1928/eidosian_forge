import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def add_unrelaxed_candidate(self, candidate, **kwargs):
    """ Add an unrelaxed starting candidate. """
    gaid = self.c.write(candidate, origin='StartingCandidateUnrelaxed', relaxed=0, generation=0, extinct=0, **kwargs)
    self.c.update(gaid, gaid=gaid)
    candidate.info['confid'] = gaid