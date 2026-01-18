from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def complex_to_pair(z):
    """
    Returns a vector (x,y) given z = x + y * i.
    """
    return vector([z.real(), z.imag()])