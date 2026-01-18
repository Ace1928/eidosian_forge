import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def delete_vao(self, vao_id):
    """Safely delete a Vertex Array Object belonging to this context.

        If this context is not the current context or this method is not
        called from the main thread, its deletion will be postponed until
        this context is next made active again.

        Otherwise, this method will immediately delete the VAO via
        ``glDeleteVertexArrays``.

        :Parameters:
            `vao_id` : int
                The OpenGL name of the Vertex Array to delete.

        .. versionadded:: 2.0
        """
    if self._safe_to_operate_on():
        gl.glDeleteVertexArrays(1, gl.GLuint(vao_id))
    else:
        self.doomed_vaos.append(vao_id)