from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_greytrack(self, track):
    """Drawing element for grey background to passed Track object."""
    greytrack_bgs = []
    greytrack_labels = []
    if not track.greytrack:
        return ([], [])
    btm, ctr, top = self.track_radii[self.current_track_level]
    start, end = self._current_track_start_end()
    startangle, startcos, startsin = self.canvas_angle(start)
    endangle, endcos, endsin = self.canvas_angle(end)
    if track.start is not None or track.end is not None:
        p = ArcPath(strokeColor=track.scale_color, fillColor=None)
        greytrack_bgs.append(self._draw_arc(btm, top, startangle, endangle, colors.Color(0.96, 0.96, 0.96)))
    elif self.sweep < 1:
        greytrack_bgs.append(self._draw_arc(btm, top, 0, 2 * pi * self.sweep, colors.Color(0.96, 0.96, 0.96)))
    else:
        greytrack_bgs.append(Circle(self.xcenter, self.ycenter, ctr, strokeColor=colors.Color(0.96, 0.96, 0.96), fillColor=None, strokeWidth=top - btm))
    if track.greytrack_labels:
        labelstep = self.length // track.greytrack_labels
        for pos in range(self.start, self.end, labelstep):
            label = String(0, 0, track.name, fontName=track.greytrack_font, fontSize=track.greytrack_fontsize, fillColor=track.greytrack_fontcolor)
            theta, costheta, sintheta = self.canvas_angle(pos)
            if theta < startangle or endangle < theta:
                continue
            x, y = (self.xcenter + btm * sintheta, self.ycenter + btm * costheta)
            labelgroup = Group(label)
            labelangle = self.sweep * 2 * pi * (pos - self.start) / self.length - pi / 2
            if theta > pi:
                label.textAnchor = 'end'
                labelangle += pi
            cosA, sinA = (cos(labelangle), sin(labelangle))
            labelgroup.transform = (cosA, -sinA, sinA, cosA, x, y)
            if not self.length - x <= labelstep:
                greytrack_labels.append(labelgroup)
    return (greytrack_bgs, greytrack_labels)