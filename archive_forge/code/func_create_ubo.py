import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def create_ubo(self, index=0):
    """
        Create a new UniformBufferObject from this uniform block.

        :Parameters:
            `index` : int
                The uniform buffer index the returned UBO will bind itself to.
                By default, this is 0.

        :rtype: :py:class:`~pyglet.graphics.shader.UniformBufferObject`
        """
    if self.view_cls is None:
        self.view_cls = self._introspect_uniforms()
    return UniformBufferObject(self.view_cls, self.size, index)