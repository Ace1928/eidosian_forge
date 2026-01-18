from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
def glutCreateWindow(title):
    """Create window with given title
        
        This is the Win32-specific version that handles
        registration of an exit-function handler 
        """
    return __glutCreateWindowWithExit(title, _exitfunc)