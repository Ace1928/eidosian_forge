import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_an_unrelaxed_candidate(self):
    """ Returns a candidate ready for relaxation. """
    to_get = self.__get_ids_of_all_unrelaxed_candidates__()
    if len(to_get) == 0:
        raise ValueError('No unrelaxed candidate to return')
    a = self.__get_latest_traj_for_confid__(to_get[0])
    a.info['confid'] = to_get[0]
    if 'data' not in a.info:
        a.info['data'] = {}
    return a