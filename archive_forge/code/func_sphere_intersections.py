from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm
def sphere_intersections(z, d, trust_radius, entire_line=False):
    """Find the intersection between segment (or line) and spherical constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the ball
    ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    trust_radius : float
        Ball radius.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the ball
        ``||x|| <= trust_radius``. When ``False``, the function returns the intersection
        between the segment ``x(t) = z + t*d``, ``0 <= t <= 1``, and the ball.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the ball for
        for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line/segment
        and the sphere. On the other hand, when ``False``, there is no
        intersection.
    """
    if norm(d) == 0:
        return (0, 0, False)
    if np.isinf(trust_radius):
        if entire_line:
            ta = -np.inf
            tb = np.inf
        else:
            ta = 0
            tb = 1
        intersect = True
        return (ta, tb, intersect)
    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius ** 2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        intersect = False
        return (0, 0, intersect)
    sqrt_discriminant = np.sqrt(discriminant)
    aux = b + copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    ta, tb = sorted([ta, tb])
    if entire_line:
        intersect = True
    elif tb < 0 or ta > 1:
        intersect = False
        ta = 0
        tb = 0
    else:
        intersect = True
        ta = max(0, ta)
        tb = min(1, tb)
    return (ta, tb, intersect)