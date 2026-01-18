from ..upper_halfspace import psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import matrix # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
def r13_fixed_points_of_psl2c_matrix(m):
    """
    Given a PSL(2,C)-matrix m acting on the upper halfspace model,
    computes the corresponding (ideal) fixed points as light-like
    vectors in the hyperboloid model.
    """
    e0 = abs(m[1, 0])
    e1 = abs(m[1, 0] - m[0, 0] + m[1, 1] - m[0, 1])
    if is_RealIntervalFieldElement(e0):
        if e0.center() > e1.center():
            return _r13_fixed_points_of_psl2c_matrix(m)
    elif e0 > e1:
        return _r13_fixed_points_of_psl2c_matrix(m)
    t = matrix([[1, 0], [1, 1]], ring=m.base_ring())
    tinv = matrix([[1, 0], [-1, 1]], ring=m.base_ring())
    pts = _r13_fixed_points_of_psl2c_matrix(tinv * m * t)
    o13_t = psl2c_to_o13(t)
    return [o13_t * pt for pt in pts]