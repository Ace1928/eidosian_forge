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
def get_nnmat_string(atoms, decimals=2, mic=False):
    nnmat = get_nnmat(atoms, mic=mic)
    s = '-'.join(['{1:2.{0}f}'.format(decimals, i) for i in nnmat])
    if len(nnmat) == 1:
        return s + '-'
    return s