from ase.ga.offspring_creator import OffspringCreator
from ase import Atoms
from itertools import chain
import numpy as np
def get_vectors_below_min_dist(self, atoms):
    """Generator function that returns each vector (between atoms)
        that is shorter than the minimum distance for those atom types
        (set during the initialization in blmin)."""
    norm = np.linalg.norm
    ap = atoms.get_positions()
    an = atoms.numbers
    for i in range(len(atoms)):
        pos = atoms[i].position
        for j, d in enumerate([norm(k - pos) for k in ap[i:]]):
            if d == 0:
                continue
            min_dist = self.blmin[tuple(sorted((an[i], an[j + i])))]
            if d < min_dist:
                yield (atoms[i].position - atoms[j + i].position, min_dist)