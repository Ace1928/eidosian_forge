import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def in_simplex(self, S, v_x, A_j0=None):
    """Check if a vector v_x is in simplex `S`.

        Parameters
        ----------
        S : array_like
            Array containing simplex entries of vertices as rows
        v_x :
            A candidate vertex
        A_j0 : array, optional,
            Allows for A_j0 to be pre-calculated

        Returns
        -------
        res : boolean
            True if `v_x` is in `S`
        """
    A_11 = numpy.delete(S, 0, 0) - S[0]
    sign_det_A_11 = numpy.sign(numpy.linalg.det(A_11))
    if sign_det_A_11 == 0:
        sign_det_A_11 = -1
    if A_j0 is None:
        A_j0 = S - v_x
    for d in range(self.dim + 1):
        det_A_jj = (-1) ** d * sign_det_A_11
        sign_det_A_j0 = numpy.sign(numpy.linalg.det(numpy.delete(A_j0, d, 0)))
        if det_A_jj == sign_det_A_j0:
            continue
        else:
            return False
    return True