from ase.lattice.bravais import Bravais, reduceindex
import numpy as np
from ase.data import reference_states as _refstate
def find_directions(self, directions, miller):
    """Find missing directions and miller indices from the specified ones."""
    directions = list(directions)
    miller = list(miller)
    self.find_ortho(directions)
    self.find_ortho(miller)
    Bravais.find_directions(self, directions, miller)