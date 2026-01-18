from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def set_polylines(self, polylines, thickness=5, tension1=1.0, tension2=1.0):
    self.clear()
    self.polylines = polylines
    self.vertices = []
    self.tension1 = tension1
    self.tension2 = tension2
    self._build_curves()
    self.draw(thickness=thickness)