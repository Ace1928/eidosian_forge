from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_bar_graph(self, graph):
    """Return list of drawable elements for a bar graph.

        Arguments:
         - graph     Graph object

        """
    bar_elements = []
    data_quartiles = graph.quartiles()
    minval, maxval = (data_quartiles[0], data_quartiles[4])
    btm, ctr, top = self.track_radii[self.current_track_level]
    trackheight = 0.5 * (top - btm)
    datarange = maxval - minval
    if datarange == 0:
        datarange = trackheight
    data = graph[self.start:self.end]
    if graph.center is None:
        midval = (maxval + minval) / 2.0
    else:
        midval = graph.center
    start, end = self._current_track_start_end()
    data = intermediate_points(start, end, graph[start:end])
    if not data:
        return []
    resolution = max(midval - minval, maxval - midval)
    if resolution == 0:
        resolution = trackheight
    for pos0, pos1, val in data:
        pos0angle, pos0cos, pos0sin = self.canvas_angle(pos0)
        pos1angle, pos1cos, pos1sin = self.canvas_angle(pos1)
        barval = trackheight * (val - midval) / resolution
        if barval >= 0:
            barcolor = graph.poscolor
        else:
            barcolor = graph.negcolor
        bar_elements.append(self._draw_arc(ctr, ctr + barval, pos0angle, pos1angle, barcolor))
    return bar_elements