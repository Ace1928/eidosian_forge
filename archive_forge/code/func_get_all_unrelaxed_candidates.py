import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_all_unrelaxed_candidates(self):
    """Return all unrelaxed candidates,
        useful if they can all be evaluated quickly."""
    to_get = self.__get_ids_of_all_unrelaxed_candidates__()
    if len(to_get) == 0:
        return []
    res = []
    for confid in to_get:
        a = self.__get_latest_traj_for_confid__(confid)
        a.info['confid'] = confid
        if 'data' not in a.info:
            a.info['data'] = {}
        res.append(a)
    return res