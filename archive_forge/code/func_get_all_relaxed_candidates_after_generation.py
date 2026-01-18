import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_all_relaxed_candidates_after_generation(self, gen):
    """ Returns all candidates that have been relaxed up to
            and including the specified generation
        """
    q = 'relaxed=1,extinct=0,generation<={0}'
    entries = self.c.select(q.format(gen))
    trajs = []
    for v in entries:
        t = self.get_atoms(id=v.id)
        t.info['confid'] = v.gaid
        t.info['relax_id'] = v.id
        trajs.append(t)
    trajs.sort(key=lambda x: get_raw_score(x), reverse=True)
    return trajs