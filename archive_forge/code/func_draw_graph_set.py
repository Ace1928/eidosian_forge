from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_graph_set(self, set):
    """Return list of graph elements and list of their labels.

        Arguments:
         - set       GraphSet object

        """
    elements = []
    style_methods = {'line': self.draw_line_graph, 'heat': self.draw_heat_graph, 'bar': self.draw_bar_graph}
    for graph in set.get_graphs():
        elements += style_methods[graph.style](graph)
    return (elements, [])