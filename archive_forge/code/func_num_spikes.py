import math
from abc import ABC, abstractmethod
import pyglet
from pyglet.gl import GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_TRIANGLES
from pyglet.gl import glBlendFunc, glEnable, glDisable
from pyglet.graphics import Batch, Group
from pyglet.math import Vec2
@num_spikes.setter
def num_spikes(self, value):
    self._num_spikes = value
    self._update_vertices()