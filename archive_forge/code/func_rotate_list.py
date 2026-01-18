import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def rotate_list(L, s):
    """Rotate the list, putting L[s] into index 0."""
    n = len(L)
    return [L[(i + s) % n] for i in range(n)]