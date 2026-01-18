from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_all_element_mutations(self, a):
    """Get all possible mutations for the supplied atoms object given
        the element pools."""
    muts = []
    symset = set(a.get_chemical_symbols())
    for sym in symset:
        for pool in self.element_pools:
            if sym in pool:
                muts.extend([(sym, s) for s in pool if s not in symset])
    return muts