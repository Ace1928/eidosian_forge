from collections import namedtuple
import math
import warnings
@property
def column_vectors(self):
    """The values of the transform as three 2D column vectors"""
    a, b, c, d, e, f, _, _, _ = self
    return ((a, d), (b, e), (c, f))