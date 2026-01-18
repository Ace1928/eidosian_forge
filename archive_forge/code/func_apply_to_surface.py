from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
def apply_to_surface(self, verts, u_set, v_set, set_len=None, inc_pos=None):
    """
        Apply this color scheme to a
        set of vertices over two
        independent variables u and v.
        """
    bounds = create_bounds()
    cverts = []
    if callable(set_len):
        set_len(len(u_set) * len(v_set) * 2)
    for _u in range(len(u_set)):
        column = []
        for _v in range(len(v_set)):
            if verts[_u][_v] is None:
                column.append(None)
            else:
                x, y, z = verts[_u][_v]
                u, v = (u_set[_u], v_set[_v])
                c = self(x, y, z, u, v)
                if c is not None:
                    c = list(c)
                    update_bounds(bounds, c)
                column.append(c)
            if callable(inc_pos):
                inc_pos()
        cverts.append(column)
    for _u in range(len(u_set)):
        for _v in range(len(v_set)):
            if cverts[_u][_v] is not None:
                for _c in range(3):
                    cverts[_u][_v][_c] = rinterpolate(bounds[_c][0], bounds[_c][1], cverts[_u][_v][_c])
                cverts[_u][_v] = self.gradient(*cverts[_u][_v])
            if callable(inc_pos):
                inc_pos()
    return cverts