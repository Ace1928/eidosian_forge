import sys
import numpy as np
from matplotlib import _api
def get_masked_triangles(self):
    """
        Return an array of triangles taking the mask into account.
        """
    if self.mask is not None:
        return self.triangles[~self.mask]
    else:
        return self.triangles