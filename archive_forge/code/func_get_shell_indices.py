import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
@classmethod
def get_shell_indices(cls, atoms, atomic_conf, min_ratio, recurs=0):
    """Recursive function that returns the indices in the surface
        subject to the min_ratio constraint. The indices are found from
        the supplied atomic configuration."""
    elements = list(set([atoms[i].symbol for subl in atomic_conf for i in subl]))
    shell = [i for subl in atomic_conf[-1 - recurs:] for i in subl]
    while len(shell) < 1:
        recurs += 1
        shell = [i for subl in atomic_conf[-1 - recurs:] for i in subl]
    for elem in elements:
        ratio = len([i for i in shell if atoms[i].symbol == elem]) / float(len(shell))
        if ratio < min_ratio:
            return COM2surfPermutation.get_shell_indices(atoms, atomic_conf, min_ratio, recurs + 1)
    return shell