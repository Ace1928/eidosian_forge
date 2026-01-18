import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def add_relaxed_step(self, a, find_neighbors=None, perform_parametrization=None):
    """After a candidate is relaxed it must be marked
        as such. Use this function if the candidate has already been in the
        database in an unrelaxed version, i.e. add_unrelaxed_candidate has
        been used.

        Neighbor list and parametrization parameters to screen
        candidates before relaxation can be added. Default is not to use.
        """
    err_msg = "raw_score not put in atoms.info['key_value_pairs']"
    assert 'raw_score' in a.info['key_value_pairs'], err_msg
    gaid = a.info['confid']
    if 'generation' not in a.info['key_value_pairs']:
        g = self.get_generation_number()
        a.info['key_value_pairs']['generation'] = g
    if find_neighbors is not None:
        set_neighbor_list(a, find_neighbors(a))
    if perform_parametrization is not None:
        set_parametrization(a, perform_parametrization(a))
    relax_id = self.c.write(a, relaxed=1, gaid=gaid, key_value_pairs=a.info['key_value_pairs'], data=a.info['data'])
    a.info['relax_id'] = relax_id