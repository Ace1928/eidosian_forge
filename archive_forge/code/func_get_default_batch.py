import ctypes
import weakref
import pyglet
from pyglet.gl import *
from pyglet.graphics import shader, vertexdomain
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import BufferObject
def get_default_batch():
    try:
        return pyglet.gl.current_context.pyglet_graphics_default_batch
    except AttributeError:
        pyglet.gl.current_context.pyglet_graphics_default_batch = Batch()
        return pyglet.gl.current_context.pyglet_graphics_default_batch