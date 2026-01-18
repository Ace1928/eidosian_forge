import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
def max_distance_rectangle(self, other, p=2.0):
    """
        Compute the maximum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float, optional
            Input.

        """
    return minkowski_distance(0, np.maximum(self.maxes - other.mins, other.maxes - self.mins), p)