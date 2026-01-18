from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_possible_mutations(self, a):
    unique_syms, comp = np.unique(sorted(a.get_chemical_symbols()), return_counts=True)
    min_num = min([i for i in np.ravel(list(self.allowed_compositions)) if i > 0])
    muts = set()
    for i, n in enumerate(comp):
        if n != 0:
            muts.add((unique_syms[i], n))
        if n % min_num >= 0:
            for j in range(1, n // min_num):
                muts.add((unique_syms[i], min_num * j))
    return list(muts)