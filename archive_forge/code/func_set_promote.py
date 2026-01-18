from numba.core.typeconv import castgraph, Conversion
from numba.core import types
def set_promote(self, fromty, toty):
    self.set_compatible(fromty, toty, Conversion.promote)