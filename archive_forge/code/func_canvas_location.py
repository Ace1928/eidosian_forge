from reportlab.graphics.shapes import Drawing, Line, String, Group, Polygon
from reportlab.lib import colors
from ._AbstractDrawer import AbstractDrawer, draw_box, draw_arrow
from ._AbstractDrawer import draw_cut_corner_box, _stroke_and_fill_colors
from ._AbstractDrawer import intermediate_points, angle2trig, deduplicate
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import ceil
def canvas_location(self, base):
    """Canvas location of a base on the genome.

        Arguments:
         - base      The base number on the genome sequence

        Returns the x-coordinate and fragment number of a base on the
        genome sequence, in the context of the current drawing setup
        """
    base = int(base - self.start)
    fragment = int(base / self.fragment_bases)
    if fragment < 1:
        base_offset = base
        fragment = 0
    elif fragment >= self.fragments:
        fragment = self.fragments - 1
        base_offset = self.fragment_bases
    else:
        base_offset = base % self.fragment_bases
    assert fragment < self.fragments, (base, self.start, self.end, self.length, self.fragment_bases)
    x_offset = self.pagewidth * base_offset / self.fragment_bases
    return (fragment, x_offset)