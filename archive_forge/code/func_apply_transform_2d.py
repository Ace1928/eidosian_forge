import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
def apply_transform_2d(self, transform):
    """Apply a transformation on x, y, z, px, py, pz,
        ox, oy, oz, dx, dy, dz.
        """
    self.x, self.y = self.pos = transform(self.x, self.y)
    self.px, self.py = transform(self.px, self.py)
    self.ox, self.oy = transform(self.ox, self.oy)
    self.dx = self.x - self.px
    self.dy = self.y - self.py