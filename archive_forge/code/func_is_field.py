from ..sage_helper import _within_sage
from snappy.number import SnapPyNumbers, Number
from itertools import chain
from ..pari import pari, PariError
from .fundamental_polyhedron import Infinity
def is_field(R):
    return isinstance(R, SnapPyNumbers)