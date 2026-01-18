import itertools
import numpy as np
from ase.geometry.dimensionality.disjoint_set import DisjointSet
def insert_bond(self, i, j, offset):
    """Inserts a bond into the component graph, both in the single cell and
        each of the n^3 subcells of the supercell.

        Parameters:

        i: int           The index of the first atom.
        n: int           The index of the second atom.
        offset: tuple    The cell offset of the second atom.
        """
    nbr_cells = (self.cells + offset) % self.n
    nbr_offsets = self.num_atoms * np.dot(self.m, nbr_cells.T)
    self.gsingle.union(i, j)
    for a, b in zip(self.offsets, nbr_offsets):
        self.gsuper.union(a + i, b + j)
        self.gsuper.union(b + i, a + j)