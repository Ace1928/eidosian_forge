from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def permute2(atoms, rng=np.random):
    i1 = rng.choice(range(len(atoms)))
    sym1 = atoms[i1].symbol
    i2 = rng.choice([a.index for a in atoms if a.symbol != sym1])
    atoms[i1].symbol = atoms[i2].symbol
    atoms[i2].symbol = sym1