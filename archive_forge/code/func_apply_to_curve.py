from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
def apply_to_curve(self, verts, u_set, set_len=None, inc_pos=None):
    """
        Apply this color scheme to a
        set of vertices over a single
        independent variable u.
        """
    bounds = create_bounds()
    cverts = []
    if callable(set_len):
        set_len(len(u_set) * 2)
    for _u in range(len(u_set)):
        if verts[_u] is None:
            cverts.append(None)
        else:
            x, y, z = verts[_u]
            u, v = (u_set[_u], None)
            c = self(x, y, z, u, v)
            if c is not None:
                c = list(c)
                update_bounds(bounds, c)
            cverts.append(c)
        if callable(inc_pos):
            inc_pos()
    for _u in range(len(u_set)):
        if cverts[_u] is not None:
            for _c in range(3):
                cverts[_u][_c] = rinterpolate(bounds[_c][0], bounds[_c][1], cverts[_u][_c])
            cverts[_u] = self.gradient(*cverts[_u])
        if callable(inc_pos):
            inc_pos()
    return cverts