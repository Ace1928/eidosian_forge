from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_scale(self, track):
    """Return list of elements in the scale and list of their labels.

        Arguments:
         - track     Track object

        """
    scale_elements = []
    scale_labels = []
    if not track.scale:
        return ([], [])
    btm, ctr, top = self.track_radii[self.current_track_level]
    trackheight = top - ctr
    start, end = self._current_track_start_end()
    if track.start is not None or track.end is not None:
        p = ArcPath(strokeColor=track.scale_color, fillColor=None)
        startangle, startcos, startsin = self.canvas_angle(start)
        endangle, endcos, endsin = self.canvas_angle(end)
        p.addArc(self.xcenter, self.ycenter, ctr, 90 - endangle * 180 / pi, 90 - startangle * 180 / pi)
        scale_elements.append(p)
        del p
        x0, y0 = (self.xcenter + btm * startsin, self.ycenter + btm * startcos)
        x1, y1 = (self.xcenter + top * startsin, self.ycenter + top * startcos)
        scale_elements.append(Line(x0, y0, x1, y1, strokeColor=track.scale_color))
        x0, y0 = (self.xcenter + btm * endsin, self.ycenter + btm * endcos)
        x1, y1 = (self.xcenter + top * endsin, self.ycenter + top * endcos)
        scale_elements.append(Line(x0, y0, x1, y1, strokeColor=track.scale_color))
    elif self.sweep < 1:
        p = ArcPath(strokeColor=track.scale_color, fillColor=None)
        p.addArc(self.xcenter, self.ycenter, ctr, startangledegrees=90 - 360 * self.sweep, endangledegrees=90)
        scale_elements.append(p)
        del p
        x0, y0 = (self.xcenter, self.ycenter + btm)
        x1, y1 = (self.xcenter, self.ycenter + top)
        scale_elements.append(Line(x0, y0, x1, y1, strokeColor=track.scale_color))
        alpha = 2 * pi * self.sweep
        x0, y0 = (self.xcenter + btm * sin(alpha), self.ycenter + btm * cos(alpha))
        x1, y1 = (self.xcenter + top * sin(alpha), self.ycenter + top * cos(alpha))
        scale_elements.append(Line(x0, y0, x1, y1, strokeColor=track.scale_color))
    else:
        scale_elements.append(Circle(self.xcenter, self.ycenter, ctr, strokeColor=track.scale_color, fillColor=None))
    start, end = self._current_track_start_end()
    if track.scale_ticks:
        ticklen = track.scale_largeticks * trackheight
        tickiterval = int(track.scale_largetick_interval)
        for tickpos in range(tickiterval * (self.start // tickiterval), int(self.end), tickiterval):
            if tickpos <= start or end <= tickpos:
                continue
            tick, label = self.draw_tick(tickpos, ctr, ticklen, track, track.scale_largetick_labels)
            scale_elements.append(tick)
            if label is not None:
                scale_labels.append(label)
        ticklen = track.scale_smallticks * trackheight
        tickiterval = int(track.scale_smalltick_interval)
        for tickpos in range(tickiterval * (self.start // tickiterval), int(self.end), tickiterval):
            if tickpos <= start or end <= tickpos:
                continue
            tick, label = self.draw_tick(tickpos, ctr, ticklen, track, track.scale_smalltick_labels)
            scale_elements.append(tick)
            if label is not None:
                scale_labels.append(label)
    startangle, startcos, startsin = self.canvas_angle(start)
    endangle, endcos, endsin = self.canvas_angle(end)
    if track.axis_labels:
        for set in track.get_sets():
            if set.__class__ is GraphSet:
                for n in range(7):
                    angle = n * 1.0471975511965976
                    if angle < startangle or endangle < angle:
                        continue
                    ticksin, tickcos = (sin(angle), cos(angle))
                    x0, y0 = (self.xcenter + btm * ticksin, self.ycenter + btm * tickcos)
                    x1, y1 = (self.xcenter + top * ticksin, self.ycenter + top * tickcos)
                    scale_elements.append(Line(x0, y0, x1, y1, strokeColor=track.scale_color))
                    graph_label_min = []
                    graph_label_max = []
                    graph_label_mid = []
                    for graph in set.get_graphs():
                        quartiles = graph.quartiles()
                        minval, maxval = (quartiles[0], quartiles[4])
                        if graph.center is None:
                            midval = (maxval + minval) / 2.0
                            graph_label_min.append(f'{minval:.3f}')
                            graph_label_max.append(f'{maxval:.3f}')
                            graph_label_mid.append(f'{midval:.3f}')
                        else:
                            diff = max(graph.center - minval, maxval - graph.center)
                            minval = graph.center - diff
                            maxval = graph.center + diff
                            midval = graph.center
                            graph_label_mid.append(f'{midval:.3f}')
                            graph_label_min.append(f'{minval:.3f}')
                            graph_label_max.append(f'{maxval:.3f}')
                    xmid, ymid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                    for limit, x, y in [(graph_label_min, x0, y0), (graph_label_max, x1, y1), (graph_label_mid, xmid, ymid)]:
                        label = String(0, 0, ';'.join(limit), fontName=track.scale_font, fontSize=track.scale_fontsize, fillColor=track.scale_color)
                        label.textAnchor = 'middle'
                        labelgroup = Group(label)
                        labelgroup.transform = (tickcos, -ticksin, ticksin, tickcos, x, y)
                        scale_labels.append(labelgroup)
    return (scale_elements, scale_labels)