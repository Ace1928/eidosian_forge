import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
@classmethod
def from_power_basis(cls, pp, extrapolate=None):
    """
        Construct a piecewise polynomial in Bernstein basis
        from a power basis polynomial.

        Parameters
        ----------
        pp : PPoly
            A piecewise polynomial in the power basis
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
    if not isinstance(pp, PPoly):
        raise TypeError('.from_power_basis only accepts PPoly instances. Got %s instead.' % type(pp))
    dx = cupy.diff(pp.x)
    k = pp.c.shape[0] - 1
    rest = (None,) * (pp.c.ndim - 2)
    c = cupy.zeros_like(pp.c)
    for a in range(k + 1):
        factor = pp.c[a] / _comb(k, k - a) * dx[(slice(None),) + rest] ** (k - a)
        for j in range(k - a, k + 1):
            c[j] += factor * _comb(j, k - a)
    if extrapolate is None:
        extrapolate = pp.extrapolate
    return cls.construct_fast(c, pp.x, extrapolate, pp.axis)