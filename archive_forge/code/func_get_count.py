import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def get_count(self):
    """
        The number of leaf nodes (original observations) belonging to
        the cluster node nd. If the target node is a leaf, 1 is
        returned.

        Returns
        -------
        get_count : int
            The number of leaf nodes below the target node.

        """
    return self.count