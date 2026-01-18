import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def delete_shader(self, shader_id):
    """Safely delete a Shader belonging to this context's object space.

        This method behaves similarly to `delete_texture`, though for
        ``glDeleteShader`` instead of ``glDeleteTextures``.

        :Parameters:
            `shader_id` : int
                The OpenGL name of the Shader to delete.

        .. versionadded:: 2.0.10
        """
    if self._safe_to_operate_on_object_space():
        gl.glDeleteShader(gl.GLuint(shader_id))
    else:
        self.object_space.doomed_shaders.append(shader_id)