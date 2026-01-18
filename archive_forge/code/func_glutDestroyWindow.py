from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
def glutDestroyWindow(window):
    """Want to destroy the window, we need to do some cleanup..."""
    context = 0
    try:
        GLUT.glutSetWindow(window)
        context = contextdata.getContext()
        result = contextdata.cleanupContext(context)
        _log.info('Cleaning up context data for window %s: %s', window, result)
    except Exception as err:
        _log.error('Error attempting to clean up context data for GLUT window %s: %s', window, result)
    return _base_glutDestroyWindow(window)