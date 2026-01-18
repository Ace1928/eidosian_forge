from math import prod
import warnings
import numpy as np
from numpy import (array, transpose, searchsorted, atleast_1d, atleast_2d,
import scipy.special as spec
from scipy.special import comb
from . import _fitpack_py
from . import dfitpack
from ._polyint import _Interpolator1D
from . import _ppoly
from .interpnd import _ndim_coords_from_arrays
from ._bsplines import make_interp_spline, BSpline
import itertools  # noqa: F401
def integrate_1d(self, a, b, axis, extrapolate=None):
    """
        Compute NdPPoly representation for one dimensional definite integral

        The result is a piecewise polynomial representing the integral:

        .. math::

           p(y, z, ...) = \\int_a^b dx\\, p(x, y, z, ...)

        where the dimension integrated over is specified with the
        `axis` parameter.

        Parameters
        ----------
        a, b : float
            Lower and upper bound for integration.
        axis : int
            Dimension over which to compute the 1-D integrals
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : NdPPoly or array-like
            Definite integral of the piecewise polynomial over [a, b].
            If the polynomial was 1D, an array is returned,
            otherwise, an NdPPoly object.

        """
    if extrapolate is None:
        extrapolate = self.extrapolate
    else:
        extrapolate = bool(extrapolate)
    ndim = len(self.x)
    axis = int(axis) % ndim
    c = self.c
    swap = list(range(c.ndim))
    swap.insert(0, swap[axis])
    del swap[axis + 1]
    swap.insert(1, swap[ndim + axis])
    del swap[ndim + axis + 1]
    c = c.transpose(swap)
    p = PPoly.construct_fast(c.reshape(c.shape[0], c.shape[1], -1), self.x[axis], extrapolate=extrapolate)
    out = p.integrate(a, b, extrapolate=extrapolate)
    if ndim == 1:
        return out.reshape(c.shape[2:])
    else:
        c = out.reshape(c.shape[2:])
        x = self.x[:axis] + self.x[axis + 1:]
        return self.construct_fast(c, x, extrapolate=extrapolate)