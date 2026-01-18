from .simplex import *
from .corner import Corner
from .arrow import Arrow
from .perm4 import Perm4
import sys
def self_opposite(self):
    count = 0
    for corner in self.Corners:
        if corner.Tetrahedron.Class[comp(corner.Subsimplex)] == self:
            count = count + 1
    return count / 2