import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def refine_all(self, centroids=True):
    """Refine the entire domain of the current complex."""
    try:
        self.triangulated_vectors
        tvs = copy.copy(self.triangulated_vectors)
        for i, vp in enumerate(tvs):
            self.rls = self.refine_local_space(*vp, bounds=self.bounds)
            for i in self.rls:
                i
    except AttributeError as ae:
        if str(ae) == "'Complex' object has no attribute 'triangulated_vectors'":
            self.triangulate(symmetry=self.symmetry, centroid=centroids)
        else:
            raise
    return