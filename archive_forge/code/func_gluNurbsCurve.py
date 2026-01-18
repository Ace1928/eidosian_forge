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
@_lazy(_simple.gluNurbsCurve)
def gluNurbsCurve(baseFunction, nurb, knots, control, type):
    """Pythonic version of gluNurbsCurve

    Calculates knotCount, stride, and order automatically
    """
    knots = arrays.GLfloatArray.asArray(knots)
    knotCount = arrays.GLfloatArray.arraySize(knots)
    control = arrays.GLfloatArray.asArray(control)
    try:
        length, step = arrays.GLfloatArray.dimensions(control)
    except ValueError as err:
        raise error.GLUError('Need a 2-dimensional control array')
    order = knotCount - length
    if _configflags.ERROR_CHECKING:
        checkOrder(order, knotCount, 'order of NURBS curve')
        checkKnots(knots, 'knots of NURBS curve')
    return baseFunction(nurb, knotCount, knots, step, control, order, type)