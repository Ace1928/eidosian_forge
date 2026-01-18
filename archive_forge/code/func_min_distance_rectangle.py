import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
def min_distance_rectangle(self, other, p=2.0):
    """
        Compute the minimum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float
            Input.

        """
    return minkowski_distance(0, np.maximum(0, np.maximum(self.mins - other.maxes, other.mins - self.maxes)), p)