from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_symbols_to_use(self, syms):
    """Get the symbols to use for the offspring candidate. The returned
        list of symbols will respect self.allowed_compositions"""
    if self.allowed_compositions is None:
        return syms
    unique_syms, counts = np.unique(syms, return_counts=True)
    comp, unique_syms = zip(*sorted(zip(counts, unique_syms), reverse=True))
    for cc in self.allowed_compositions:
        comp += (0,) * (len(cc) - len(comp))
        if comp == tuple(sorted(cc)):
            return syms
    comp_diff = self.get_closest_composition_diff(comp)
    to_add, to_rem = get_add_remove_lists(**dict(zip(unique_syms, comp_diff)))
    for add, rem in zip(to_add, to_rem):
        tbc = [i for i in range(len(syms)) if syms[i] == rem]
        ai = self.rng.choice(tbc)
        syms[ai] = add
    return syms