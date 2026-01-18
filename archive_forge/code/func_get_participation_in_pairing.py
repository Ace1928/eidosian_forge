import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_participation_in_pairing(self):
    """ Get information about how many direct
            offsprings each candidate has, and which specific
            pairings have been made. This information is used
            for the extended fitness calculation described in
            L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
        """
    entries = self.c.select(pairing=1)
    frequency = dict()
    pairs = []
    for e in entries:
        c1, c2 = e.data['parents']
        pairs.append(tuple(sorted([c1, c2])))
        if c1 not in frequency.keys():
            frequency[c1] = 0
        frequency[c1] += 1
        if c2 not in frequency.keys():
            frequency[c2] = 0
        frequency[c2] += 1
    return (frequency, pairs)