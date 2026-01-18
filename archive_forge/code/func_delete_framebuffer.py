import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def delete_framebuffer(self, fbo_id):
    """Safely delete a Framebuffer Object belonging to this context.

        This method behaves similarly to `delete_vao`, though for
        ``glDeleteFramebuffers`` instead of ``glDeleteVertexArrays``.

        :Parameters:
            `fbo_id` : int
                The OpenGL name of the Framebuffer Object to delete.

        .. versionadded:: 2.0.10
        """
    if self._safe_to_operate_on():
        gl.glDeleteFramebuffers(1, gl.GLuint(fbo_id))
    else:
        self.doomed_framebuffers.append(fbo_id)