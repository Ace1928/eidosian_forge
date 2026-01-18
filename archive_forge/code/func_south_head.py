from .simplex import *
from .perm4 import Perm4, inv
def south_head(self):
    return self.Tetrahedron.Class[self.head() | OppTail[self.tail(), self.head()]]