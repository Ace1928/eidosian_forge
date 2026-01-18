from OpenGL.raw import GLU as _simple
from OpenGL import platform, converters, wrapper
from OpenGL.GLU import glustruct
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL import arrays, error
import ctypes
import weakref
from OpenGL.platform import PLATFORM
import OpenGL
from OpenGL import _configflags
def checkKnots(knots, name):
    """Check that knots are in ascending order"""
    if len(knots):
        knot = knots[0]
        for next in knots[1:]:
            if next < knot:
                raise error.GLUError('%s has decreasing knot %s after %s' % (name, next, knot))