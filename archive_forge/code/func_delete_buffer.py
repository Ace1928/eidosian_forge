import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def delete_buffer(self, buffer_id):
    """Safely delete a Buffer object belonging to this context's object
        space.

        This method behaves similarly to `delete_texture`, though for
        ``glDeleteBuffers`` instead of ``glDeleteTextures``.

        :Parameters:
            `buffer_id` : int
                The OpenGL name of the buffer to delete.

        .. versionadded:: 1.1
        """
    if self._safe_to_operate_on_object_space():
        gl.glDeleteBuffers(1, gl.GLuint(buffer_id))
    else:
        self.object_space.doomed_buffers.append(buffer_id)