from collections import OrderedDict
from ... import sage_helper
def glued_to(self, side):
    for sides in ([S for S in self.sides], [-S for S in self.sides]):
        if side in sides:
            sides.remove(side)
            return sides[0]
    raise IndexError('Given side does not appear in this edge')