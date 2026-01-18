import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def is_leaf(self):
    """
        Return True if the target node is a leaf.

        Returns
        -------
        leafness : bool
            True if the target node is a leaf node.

        """
    return self.left is None