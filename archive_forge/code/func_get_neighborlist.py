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
def get_neighborlist(atoms, dx=0.2, no_count_types=None):
    """Method to get the a dict with list of neighboring
    atoms defined as the two covalent radii + fixed distance.
    Option added to remove neighbors between defined atom types.
    """
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    if no_count_types is None:
        no_count_types = []
    conn = {}
    for atomi in atoms:
        conn_this_atom = []
        for atomj in atoms:
            if atomi.index != atomj.index:
                if atomi.number not in no_count_types:
                    if atomj.number not in no_count_types:
                        d = get_mic_distance(atomi.position, atomj.position, cell, pbc)
                        cri = covalent_radii[atomi.number]
                        crj = covalent_radii[atomj.number]
                        d_max = crj + cri + dx
                        if d < d_max:
                            conn_this_atom.append(atomj.index)
        conn[atomi.index] = conn_this_atom
    return conn