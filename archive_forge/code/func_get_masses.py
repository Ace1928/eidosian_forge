import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
def get_masses(self):
    m = self.atoms.get_masses()
    masses = np.zeros(self.n)
    for i in range(self.n):
        indices = np.where(self.tags == self.unique_tags[i])
        masses[i] = np.sum(m[indices])
    return masses