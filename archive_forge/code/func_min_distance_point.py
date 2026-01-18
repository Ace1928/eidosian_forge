import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
def min_distance_point(self, x, p=2.0):
    """
        Return the minimum distance between input and points in the
        hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input.
        p : float, optional
            Input.

        """
    return minkowski_distance(0, np.maximum(0, np.maximum(self.mins - x, x - self.maxes)), p)