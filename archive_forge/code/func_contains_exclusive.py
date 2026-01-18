from param.parameterized import get_occupied_slots
from .util import datetime_types
def contains_exclusive(self, x, y):
    """
        Return True if the given point is contained within the
        bounding box, where the bottom and right boundaries are
        considered exclusive.
        """
    left, bottom, right, top = self._aarect.lbrt()
    return left <= x < right and bottom < y <= top