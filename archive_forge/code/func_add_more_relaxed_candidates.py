import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def add_more_relaxed_candidates(self, a_list):
    """Add more relaxed candidates quickly"""
    for a in a_list:
        try:
            a.info['key_value_pairs']['raw_score']
        except KeyError:
            print("raw_score not put in atoms.info['key_value_pairs']")
    g = self.get_generation_number()
    next_id = self.get_next_id()
    with self.c as con:
        for j, a in enumerate(a_list):
            if 'generation' not in a.info['key_value_pairs']:
                a.info['key_value_pairs']['generation'] = g
            gaid = next_id + j
            relax_id = con.write(a, relaxed=1, gaid=gaid, key_value_pairs=a.info['key_value_pairs'], data=a.info['data'])
            assert gaid == relax_id
            a.info['confid'] = relax_id
            a.info['relax_id'] = relax_id