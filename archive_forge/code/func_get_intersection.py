from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def get_intersection(cx1, cy1, cos_t1, sin_t1, cx2, cy2, cos_t2, sin_t2):
    """
    Return the intersection between the line through (*cx1*, *cy1*) at angle
    *t1* and the line through (*cx2*, *cy2*) at angle *t2*.
    """
    line1_rhs = sin_t1 * cx1 - cos_t1 * cy1
    line2_rhs = sin_t2 * cx2 - cos_t2 * cy2
    a, b = (sin_t1, -cos_t1)
    c, d = (sin_t2, -cos_t2)
    ad_bc = a * d - b * c
    if abs(ad_bc) < 1e-12:
        raise ValueError('Given lines do not intersect. Please verify that the angles are not equal or differ by 180 degrees.')
    a_, b_ = (d, -b)
    c_, d_ = (-c, a)
    a_, b_, c_, d_ = [k / ad_bc for k in [a_, b_, c_, d_]]
    x = a_ * line1_rhs + b_ * line2_rhs
    y = c_ * line1_rhs + d_ * line2_rhs
    return (x, y)