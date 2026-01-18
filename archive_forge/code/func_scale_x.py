import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
@scale_x.setter
def scale_x(self, scale_x):
    self._scale_x = scale_x
    self._vertex_list.scale[:] = (self._scale * scale_x, self._scale * self._scale_y)