from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def add_shifts(self):
    shifts = []
    for i in range(self.Size):
        shifts += [self.Coefficients[i] * w for w in QuadShifts[self.Quadtypes[i]]]
    self.Shifts = shifts