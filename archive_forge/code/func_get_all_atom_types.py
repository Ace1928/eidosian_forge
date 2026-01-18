import os
import time
import math
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import write, read
from ase.geometry.cell import cell_to_cellpar
from ase.data import covalent_radii
from ase.ga import get_neighbor_list
def get_all_atom_types(slab, atom_numbers_to_optimize):
    """Utility method used to extract all unique atom types
    from the atoms object slab and the list of atomic numbers
    atom_numbers_to_optimize.
    """
    from_slab = list(set(slab.numbers))
    from_top = list(set(atom_numbers_to_optimize))
    from_slab.extend(from_top)
    return list(set(from_slab))