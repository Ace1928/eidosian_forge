from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_add_remove_lists(**kwargs):
    to_add, to_rem = ([], [])
    for s, amount in kwargs.items():
        if amount > 0:
            to_add.extend([s] * amount)
        elif amount < 0:
            to_rem.extend([s] * abs(amount))
    return (to_add, to_rem)