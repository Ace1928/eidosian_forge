import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def delete_shader_program(self, program_id):
    """Safely delete a Shader Program belonging to this context's
        object space.

        This method behaves similarly to `delete_texture`, though for
        ``glDeleteProgram`` instead of ``glDeleteTextures``.

        :Parameters:
            `program_id` : int
                The OpenGL name of the Shader Program to delete.

        .. versionadded:: 2.0
        """
    if self._safe_to_operate_on_object_space():
        gl.glDeleteProgram(gl.GLuint(program_id))
    else:
        self.object_space.doomed_shader_programs.append(program_id)