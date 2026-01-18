from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_feature_set(self, set):
    """Return list of feature elements and list of labels for them."""
    feature_elements = []
    label_elements = []
    for feature in set.get_features():
        if self.is_in_bounds(feature.start) or self.is_in_bounds(feature.end):
            features, labels = self.draw_feature(feature)
            feature_elements += features
            label_elements += labels
    return (feature_elements, label_elements)