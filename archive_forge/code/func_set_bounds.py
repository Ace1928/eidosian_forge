from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def set_bounds(self, start, end):
    """Set start and end points for the drawing as a whole.

        Arguments:
         - start - The first base (or feature mark) to draw from
         - end - The last base (or feature mark) to draw to

        """
    low, high = self._parent.range()
    if start is not None and end is not None and (start > end):
        start, end = (end, start)
    if start is None or start < 0:
        start = 0
    if end is None or end < 0:
        end = high + 1
    self.start, self.end = (int(start), int(end))
    self.length = self.end - self.start + 1