import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def compress_csc(self):
    """
        Compress rows, cols, vals / summing duplicates. Sort for csc format.
        """
    _, unique, indices = np.unique(self.rows + self.n * self.cols, return_index=True, return_inverse=True)
    self.rows = self.rows[unique]
    self.cols = self.cols[unique]
    self.vals = np.bincount(indices, weights=self.vals)