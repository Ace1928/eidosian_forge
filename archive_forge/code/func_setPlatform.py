imports the variables from this location, and once that 
import occurs the flags should no longer be changed.
from OpenGL.version import __version__
import os
from OpenGL.plugins import PlatformPlugin, FormatHandler
import sys
def setPlatform(key):
    """Programatically set the platform to use for PyOpenGL

    Note: you must do this *before* you import e.g. GL.* or GLES.*
    as the extension procedure lookup is platform dependent

    The PYOPENGL_PLATFORM environment variable is likely more useful
    for a *user* choosing a platform, but in cases where the programmer
    needs to choose the platform (e.g. to allow using Pygame-GLX
    under wayland) you can call `setPlatform('glx')` to force the
    use of the glx plugin.
    """
    os.environ['PYOPENGL_PLATFORM'] = key