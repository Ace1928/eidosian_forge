from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def set_track_heights(self):
    """Initialize track heights.

        Since tracks may not be of identical heights, the bottom and top
        radius for each track is stored in a dictionary - self.track_radii,
        keyed by track number
        """
    bot_track = min(min(self.drawn_tracks), 1)
    top_track = max(self.drawn_tracks)
    trackunit_sum = 0
    trackunits = {}
    heightholder = 0
    for track in range(bot_track, top_track + 1):
        try:
            trackheight = self._parent[track].height
        except Exception:
            trackheight = 1
        trackunit_sum += trackheight
        trackunits[track] = (heightholder, heightholder + trackheight)
        heightholder += trackheight
    max_radius = 0.5 * min(self.pagewidth, self.pageheight)
    trackunit_height = max_radius * (1 - self.circle_core) / trackunit_sum
    track_core = max_radius * self.circle_core
    self.track_radii = {}
    track_crop = trackunit_height * (1 - self.track_size) / 2.0
    for track in trackunits:
        top = trackunits[track][1] * trackunit_height - track_crop + track_core
        btm = trackunits[track][0] * trackunit_height + track_crop + track_core
        ctr = btm + (top - btm) / 2.0
        self.track_radii[track] = (btm, ctr, top)