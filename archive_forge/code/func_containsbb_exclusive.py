from param.parameterized import get_occupied_slots
from .util import datetime_types
def containsbb_exclusive(self, x):
    """
        Returns true if the given BoundingBox x is contained within the
        bounding box, where at least one of the boundaries of the box has
        to be exclusive.
        """
    left, bottom, right, top = self.aarect().lbrt()
    leftx, bottomx, rightx, topx = x.aarect().lbrt()
    return left <= leftx and bottom <= bottomx and (right >= rightx) and (top >= topx) and (not (left == leftx and bottom == bottomx and (right == rightx) and (top == topx)))