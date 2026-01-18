from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_tick(self, tickpos, ctr, ticklen, track, draw_label):
    """Return drawing element for a tick on the scale.

        Arguments:
         - tickpos   Int, position of the tick on the sequence
         - ctr       Float, Y co-ord of the center of the track
         - ticklen   How long to draw the tick
         - track     Track, the track the tick is drawn on
         - draw_label    Boolean, write the tick label?

        """
    tickangle, tickcos, ticksin = self.canvas_angle(tickpos)
    x0, y0 = (self.xcenter + ctr * ticksin, self.ycenter + ctr * tickcos)
    x1, y1 = (self.xcenter + (ctr + ticklen) * ticksin, self.ycenter + (ctr + ticklen) * tickcos)
    tick = Line(x0, y0, x1, y1, strokeColor=track.scale_color)
    if draw_label:
        if track.scale_format == 'SInt':
            if tickpos >= 1000000:
                tickstring = str(tickpos // 1000000) + ' Mbp'
            elif tickpos >= 1000:
                tickstring = str(tickpos // 1000) + ' Kbp'
            else:
                tickstring = str(tickpos)
        else:
            tickstring = str(tickpos)
        label = String(0, 0, tickstring, fontName=track.scale_font, fontSize=track.scale_fontsize, fillColor=track.scale_color)
        if tickangle > pi:
            label.textAnchor = 'end'
        labelgroup = Group(label)
        labelgroup.transform = (1, 0, 0, 1, x1, y1)
    else:
        labelgroup = None
    return (tick, labelgroup)