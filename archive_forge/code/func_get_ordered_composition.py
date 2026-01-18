from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_ordered_composition(syms, pools=None):
    if pools is None:
        pool_index = dict(((sym, 0) for sym in set(syms)))
    else:
        pool_index = {}
        for i, pool in enumerate(pools):
            if isinstance(pool, str):
                pool_index[pool] = i
            else:
                for sym in set(syms):
                    if sym in pool:
                        pool_index[sym] = i
    syms = [(sym, pool_index[sym], c) for sym, c in zip(*np.unique(syms, return_counts=True))]
    unique_syms, pn, comp = zip(*sorted(syms, key=lambda k: (k[1] - k[2], k[0])))
    return (unique_syms, pn, comp)