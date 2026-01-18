from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def finalize_individual(self, indi):
    atoms_string = ''.join(indi.get_chemical_symbols())
    indi.info['key_value_pairs']['atoms_string'] = atoms_string
    return OffspringCreator.finalize_individual(self, indi)