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
def gather_atoms_by_tag(atoms):
    """Translates same-tag atoms so that they lie 'together',
    with distance vectors as in the minimum image convention."""
    tags = atoms.get_tags()
    pos = atoms.get_positions()
    for tag in list(set(tags)):
        indices = np.where(tags == tag)[0]
        if len(indices) == 1:
            continue
        vectors = atoms.get_distances(indices[0], indices[1:], mic=True, vector=True)
        pos[indices[1:]] = pos[indices[0]] + vectors
    atoms.set_positions(pos)