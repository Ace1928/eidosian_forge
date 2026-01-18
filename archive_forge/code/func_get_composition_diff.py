from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_composition_diff(self, c1, c2):
    difflen = len(c1) - len(c2)
    if difflen > 0:
        c2 += (0,) * difflen
    return np.array(c2) - c1