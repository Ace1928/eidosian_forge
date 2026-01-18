import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
def read_used_modes(self, filename):
    """Read used modes from json file."""
    with open(filename, 'r') as fd:
        modes = json.load(fd)
        self.used_modes = {int(k): modes[k] for k in modes}
    return