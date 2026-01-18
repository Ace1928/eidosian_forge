from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def axis_aligned_extrema(self):
    """
        Return the dimension and location of the curve's interior extrema.

        The extrema are the points along the curve where one of its partial
        derivatives is zero.

        Returns
        -------
        dims : array of int
            Index :math:`i` of the partial derivative which is zero at each
            interior extrema.
        dzeros : array of float
            Of same size as dims. The :math:`t` such that :math:`d/dx_i B(t) =
            0`
        """
    n = self.degree
    if n <= 1:
        return (np.array([]), np.array([]))
    Cj = self.polynomial_coefficients
    dCj = np.arange(1, n + 1)[:, None] * Cj[1:]
    dims = []
    roots = []
    for i, pi in enumerate(dCj.T):
        r = np.roots(pi[::-1])
        roots.append(r)
        dims.append(np.full_like(r, i))
    roots = np.concatenate(roots)
    dims = np.concatenate(dims)
    in_range = np.isreal(roots) & (roots >= 0) & (roots <= 1)
    return (dims[in_range], np.real(roots)[in_range])