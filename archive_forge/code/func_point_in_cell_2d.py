from math import log10, atan2, cos, sin
from ase.build import hcp0001, fcc111, bcc111
import numpy as np
def point_in_cell_2d(point, cell, eps=1e-08):
    """This function takes a 2D slice of the cell in the XY plane and calculates
    if a point should lie in it.  This is used as a more accurate method of
    ensuring we find all of the correct cell repetitions in the root surface
    code.  The Z axis is totally ignored but for most uses this should be fine.
    """

    def tri_area(t1, t2, t3):
        t1x, t1y = t1[0:2]
        t2x, t2y = t2[0:2]
        t3x, t3y = t3[0:2]
        return abs(t1x * (t2y - t3y) + t2x * (t3y - t1y) + t3x * (t1y - t2y)) / 2
    c0 = (0, 0)
    c1 = cell[0, 0:2]
    c2 = cell[1, 0:2]
    c3 = c1 + c2
    cA = tri_area(c0, c1, c2) + tri_area(c1, c2, c3)
    pA = tri_area(point, c0, c1) + tri_area(point, c1, c2) + tri_area(point, c2, c3) + tri_area(point, c3, c0)
    return pA <= cA + eps