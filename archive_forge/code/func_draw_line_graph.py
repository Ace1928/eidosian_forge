from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_line_graph(self, graph):
    """Return line graph as list of drawable elements.

        Arguments:
         - graph     GraphData object

        """
    line_elements = []
    data_quartiles = graph.quartiles()
    minval, maxval = (data_quartiles[0], data_quartiles[4])
    btm, ctr, top = self.track_radii[self.current_track_level]
    trackheight = 0.5 * (top - btm)
    datarange = maxval - minval
    if datarange == 0:
        datarange = trackheight
    start, end = self._current_track_start_end()
    data = graph[start:end]
    if not data:
        return []
    if graph.center is None:
        midval = (maxval + minval) / 2.0
    else:
        midval = graph.center
    resolution = max(midval - minval, maxval - midval)
    pos, val = data[0]
    lastangle, lastcos, lastsin = self.canvas_angle(pos)
    posheight = trackheight * (val - midval) / resolution + ctr
    lastx = self.xcenter + posheight * lastsin
    lasty = self.ycenter + posheight * lastcos
    for pos, val in data:
        posangle, poscos, possin = self.canvas_angle(pos)
        posheight = trackheight * (val - midval) / resolution + ctr
        x = self.xcenter + posheight * possin
        y = self.ycenter + posheight * poscos
        line_elements.append(Line(lastx, lasty, x, y, strokeColor=graph.poscolor, strokeWidth=graph.linewidth))
        lastx, lasty = (x, y)
    return line_elements