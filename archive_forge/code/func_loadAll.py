import ctypes
from OpenGL import plugins
def loadAll(cls):
    """Load all OpenGL.plugins-registered FormatHandler classes
        """
    for entrypoint in plugins.FormatHandler.all():
        cls.loadPlugin(entrypoint)