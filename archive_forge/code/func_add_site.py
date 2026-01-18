import itertools
import numpy as np
from ase.utils import pbc2pbc
from ase.cell import Cell
def add_site(self, H):
    H = H.ravel()
    found = (self.visited == H).all(axis=1).any()
    self.visited = np.roll(self.visited, 1, axis=0)
    self.visited[0] = H
    return found