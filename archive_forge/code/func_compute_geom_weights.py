import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def compute_geom_weights(self):
    """
        Build the (nelems, 3) weights coeffs of _triangles angles,
        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
        """
    weights = np.zeros([np.size(self._triangles, 0), 3])
    tris_pts = self._tris_pts
    for ipt in range(3):
        p0 = tris_pts[:, ipt % 3, :]
        p1 = tris_pts[:, (ipt + 1) % 3, :]
        p2 = tris_pts[:, (ipt - 1) % 3, :]
        alpha1 = np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
        alpha2 = np.arctan2(p2[:, 1] - p0[:, 1], p2[:, 0] - p0[:, 0])
        angle = np.abs((alpha2 - alpha1) / np.pi % 1)
        weights[:, ipt] = 0.5 - np.abs(angle - 0.5)
    return weights