from ase.ga.offspring_creator import OffspringCreator
from ase import Atoms
from itertools import chain
import numpy as np
def get_shortest_dist_vector(self, atoms):
    norm = np.linalg.norm
    mind = 10000.0
    ap = atoms.get_positions()
    for i in range(len(atoms)):
        pos = atoms[i].position
        for j, d in enumerate([norm(k - pos) for k in ap[i:]]):
            if d == 0:
                continue
            if d < mind:
                mind = d
                lowpair = (i, j + i)
    return atoms[lowpair[0]].position - atoms[lowpair[1]].position