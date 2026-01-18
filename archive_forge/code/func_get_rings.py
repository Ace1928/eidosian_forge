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
def get_rings(atoms, rings=[5, 6, 7]):
    """This method return a list of the number of atoms involved
    in rings in the structures. It uses the neighbor
    list hence inherit the restriction used for neighbors.
    """
    conn = get_neighbor_list(atoms)
    if conn is None:
        conn = get_neighborlist(atoms)
    no_of_loops = [0] * 8
    for s1 in range(len(atoms)):
        for s2 in conn[s1]:
            v12 = [s1] + [s2]
            for s3 in [s for s in conn[s2] if s not in v12]:
                v13 = v12 + [s3]
                if s1 in conn[s3]:
                    no_of_loops[3] += 1
                for s4 in [s for s in conn[s3] if s not in v13]:
                    v14 = v13 + [s4]
                    if s1 in conn[s4]:
                        no_of_loops[4] += 1
                    for s5 in [s for s in conn[s4] if s not in v14]:
                        v15 = v14 + [s5]
                        if s1 in conn[s5]:
                            no_of_loops[5] += 1
                        for s6 in [s for s in conn[s5] if s not in v15]:
                            v16 = v15 + [s6]
                            if s1 in conn[s6]:
                                no_of_loops[6] += 1
                            for s7 in [s for s in conn[s6] if s not in v16]:
                                if s1 in conn[s7]:
                                    no_of_loops[7] += 1
    to_return = []
    for ring in rings:
        to_return.append(no_of_loops[ring])
    return to_return