from collections import OrderedDict
from ... import sage_helper
def is_zero_in_homology(self):
    B2 = self.surface.B2().transpose()
    r1 = B2.rank()
    r2 = matrix(list(B2) + [self.weights]).rank()
    return r1 == r2