from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_layer_comps(atoms, eps=0.01):
    lc = []
    old_z = np.inf
    for z, ind in sorted([(a.z, a.index) for a in atoms]):
        if abs(old_z - z) < eps:
            lc[-1].append(ind)
        else:
            lc.append([ind])
        old_z = z
    return lc