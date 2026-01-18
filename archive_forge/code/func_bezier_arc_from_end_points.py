import re, copy
from math import acos, ceil, copysign, cos, degrees, fabs, hypot, radians, sin, sqrt
from .shapes import Group, mmult, rotate, translate, transformPoint, Path, FILL_EVEN_ODD, _CLOSEPATH, UserNode
def bezier_arc_from_end_points(x1, y1, rx, ry, phi, fA, fS, x2, y2):
    if x1 == x2 and y1 == y2:
        return []
    if phi:
        mx = mmult(rotate(-phi), translate(-x1, -y1))
        tx2, ty2 = transformPoint(mx, (x2, y2))
        cx, cy, rx, ry, start_ang, extent = end_point_to_center_parameters(0, 0, tx2, ty2, fA, fS, rx, ry)
        bp = bezier_arc_from_centre(cx, cy, rx, ry, start_ang, extent)
        mx = mmult(translate(x1, y1), rotate(phi))
        res = []
        for x1, y1, x2, y2, x3, y3, x4, y4 in bp:
            res.append(transformPoint(mx, (x1, y1)) + transformPoint(mx, (x2, y2)) + transformPoint(mx, (x3, y3)) + transformPoint(mx, (x4, y4)))
        return res
    else:
        cx, cy, rx, ry, start_ang, extent = end_point_to_center_parameters(x1, y1, x2, y2, fA, fS, rx, ry)
        return bezier_arc_from_centre(cx, cy, rx, ry, start_ang, extent)