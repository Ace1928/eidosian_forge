from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def same_layer_comp(atoms, rng=np.random):
    unique_syms, comp = np.unique(sorted(atoms.get_chemical_symbols()), return_counts=True)
    l = get_layer_comps(atoms)
    sym_dict = dict(((s, int(np.array(c) / len(l))) for s, c in zip(unique_syms, comp)))
    for la in l:
        correct_by = sym_dict.copy()
        lcomp = dict(zip(*np.unique([atoms[i].symbol for i in la], return_counts=True)))
        for s, num in lcomp.items():
            correct_by[s] -= num
        to_add, to_rem = get_add_remove_lists(**correct_by)
        for add, rem in zip(to_add, to_rem):
            ai = rng.choice([i for i in la if atoms[i].symbol == rem])
            atoms[ai].symbol = add