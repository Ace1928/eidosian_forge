from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_track(self, track):
    """Return list of track elements and list of track labels."""
    track_elements = []
    track_labels = []
    set_methods = {FeatureSet: self.draw_feature_set, GraphSet: self.draw_graph_set}
    for set in track.get_sets():
        elements, labels = set_methods[set.__class__](set)
        track_elements += elements
        track_labels += labels
    return (track_elements, track_labels)