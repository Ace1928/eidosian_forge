from .simplex import *
from .perm4 import Perm4, inv
def linking_cycle(self):
    a = self.copy()
    cycle = []
    while 1:
        cycle.append(a.Tetrahedron.Class[OppositeEdge[a.Edge]])
        a.next()
        if a == self:
            break
    return cycle