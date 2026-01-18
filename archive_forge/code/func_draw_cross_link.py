from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_cross_link(self, cross_link):
    """Draw a cross-link between features."""
    startA = cross_link.startA
    startB = cross_link.startB
    endA = cross_link.endA
    endB = cross_link.endB
    if not self.is_in_bounds(startA) and (not self.is_in_bounds(endA)):
        return None
    if not self.is_in_bounds(startB) and (not self.is_in_bounds(endB)):
        return None
    if startA < self.start:
        startA = self.start
    if startB < self.start:
        startB = self.start
    if self.end < endA:
        endA = self.end
    if self.end < endB:
        endB = self.end
    trackobjA = cross_link._trackA(list(self._parent.tracks.values()))
    trackobjB = cross_link._trackB(list(self._parent.tracks.values()))
    assert trackobjA is not None
    assert trackobjB is not None
    if trackobjA == trackobjB:
        raise NotImplementedError
    if trackobjA.start is not None:
        if endA < trackobjA.start:
            return
        startA = max(startA, trackobjA.start)
    if trackobjA.end is not None:
        if trackobjA.end < startA:
            return
        endA = min(endA, trackobjA.end)
    if trackobjB.start is not None:
        if endB < trackobjB.start:
            return
        startB = max(startB, trackobjB.start)
    if trackobjB.end is not None:
        if trackobjB.end < startB:
            return
        endB = min(endB, trackobjB.end)
    for track_level in self._parent.get_drawn_levels():
        track = self._parent[track_level]
        if track == trackobjA:
            trackA = track_level
        if track == trackobjB:
            trackB = track_level
    if trackA == trackB:
        raise NotImplementedError
    startangleA, startcosA, startsinA = self.canvas_angle(startA)
    startangleB, startcosB, startsinB = self.canvas_angle(startB)
    endangleA, endcosA, endsinA = self.canvas_angle(endA)
    endangleB, endcosB, endsinB = self.canvas_angle(endB)
    btmA, ctrA, topA = self.track_radii[trackA]
    btmB, ctrB, topB = self.track_radii[trackB]
    if ctrA < ctrB:
        return [self._draw_arc_poly(topA, btmB, startangleA, endangleA, startangleB, endangleB, cross_link.color, cross_link.border, cross_link.flip)]
    else:
        return [self._draw_arc_poly(btmA, topB, startangleA, endangleA, startangleB, endangleB, cross_link.color, cross_link.border, cross_link.flip)]