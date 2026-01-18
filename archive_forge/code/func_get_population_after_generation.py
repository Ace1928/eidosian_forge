from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def get_population_after_generation(self, gen):
    """ Returns a copy of the population as it where
        after generation gen"""
    if self.logfile is not None:
        fd = open(self.logfile, 'r')
        gens = {}
        for l in fd:
            _, no, popul = l.split(':')
            gens[int(no)] = [int(i) for i in popul.split(',')]
        fd.close()
        return [c.copy() for c in self.all_cand[::-1] if c.info['relax_id'] in gens[gen]]
    all_candidates = [c for c in self.all_cand if c.info['key_value_pairs']['generation'] <= gen]
    cands = [all_candidates[0]]
    for b in all_candidates:
        if b not in cands:
            for a in cands:
                if self.comparator.looks_like(a, b):
                    break
            else:
                cands.append(b)
    pop = cands[:self.pop_size]
    return [a.copy() for a in pop]