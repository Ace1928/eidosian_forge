import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
def get_output_handler(self):
    """Fast-path lookup for output handler object"""
    if self.output_handler is None:
        if self.preferredOutput is not None:
            self.output_handler = self.handler_by_plugin_name(self.preferredOutput)
        if not self.output_handler:
            for preferred in self.GENERIC_OUTPUT_PREFERENCES:
                self.output_handler = self.handler_by_plugin_name(preferred)
                if self.output_handler:
                    break
        if not self.output_handler:
            raise RuntimeError('Unable to find any output handler at all (not even ctypes/numpy ones!)')
    return self.output_handler