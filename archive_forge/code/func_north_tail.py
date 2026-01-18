from .simplex import *
from .perm4 import Perm4, inv
def north_tail(self):
    return self.Tetrahedron.Class[self.tail() | OppTail[self.head(), self.tail()]]