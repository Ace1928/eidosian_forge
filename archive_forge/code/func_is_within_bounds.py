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
def is_within_bounds(self, cell):
    values = get_cell_angles_lengths(cell)
    verdict = True
    for param, bound in self.bounds.items():
        if not bound[0] <= values[param] <= bound[1]:
            verdict = False
    return verdict