import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_all_relaxed_candidates(self, only_new=False, use_extinct=False):
    """ Returns all candidates that have been relaxed.

        Parameters:

        only_new: boolean (optional)
            Used to specify only to get candidates relaxed since last
            time this function was invoked. Default: False.

        use_extinct: boolean (optional)
            Set to True if the extinct key (and mass extinction) is going
            to be used. Default: False."""
    if use_extinct:
        entries = self.c.select('relaxed=1,extinct=0', sort='-raw_score')
    else:
        entries = self.c.select('relaxed=1', sort='-raw_score')
    trajs = []
    for v in entries:
        if only_new and v.gaid in self.already_returned:
            continue
        t = self.get_atoms(id=v.id)
        t.info['confid'] = v.gaid
        t.info['relax_id'] = v.id
        trajs.append(t)
        self.already_returned.add(v.gaid)
    return trajs