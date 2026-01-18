from .simplex import *
from .perm4 import Perm4, inv
def north_head(self):
    return self.Tetrahedron.Class[self.head() | OppTail[self.head(), self.tail()]]