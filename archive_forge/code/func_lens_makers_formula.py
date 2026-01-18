from sympy.core.numbers import (Float, I, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2, cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import cancel
from sympy.series.limits import Limit
from sympy.geometry.line import Ray3D
from sympy.geometry.util import intersection
from sympy.geometry.plane import Plane
from sympy.utilities.iterables import is_sequence
from .medium import Medium
def lens_makers_formula(n_lens, n_surr, r1, r2, d=0):
    """
    This function calculates focal length of a lens.
    It follows cartesian sign convention.

    Parameters
    ==========

    n_lens : Medium or sympifiable
        Index of refraction of lens.
    n_surr : Medium or sympifiable
        Index of reflection of surrounding.
    r1 : sympifiable
        Radius of curvature of first surface.
    r2 : sympifiable
        Radius of curvature of second surface.
    d : sympifiable, optional
        Thickness of lens, default value is 0.

    Examples
    ========

    >>> from sympy.physics.optics import lens_makers_formula
    >>> from sympy import S
    >>> lens_makers_formula(1.33, 1, 10, -10)
    15.1515151515151
    >>> lens_makers_formula(1.2, 1, 10, S.Infinity)
    50.0000000000000
    >>> lens_makers_formula(1.33, 1, 10, -10, d=1)
    15.3418463277618

    """
    if isinstance(n_lens, Medium):
        n_lens = n_lens.refractive_index
    else:
        n_lens = sympify(n_lens)
    if isinstance(n_surr, Medium):
        n_surr = n_surr.refractive_index
    else:
        n_surr = sympify(n_surr)
    d = sympify(d)
    focal_length = 1 / ((n_lens - n_surr) / n_surr * (1 / r1 - 1 / r2 + (n_lens - n_surr) * d / (n_lens * r1 * r2)))
    if focal_length == zoo:
        return S.Infinity
    return focal_length