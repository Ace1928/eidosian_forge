from OpenGL.raw import GLU as _simple
from OpenGL import GL
from OpenGL.lazywrapper import lazy as _lazy
import ctypes 
@_lazy(_simple.gluProject)
def gluProject(baseFunction, objX, objY, objZ, model=None, proj=None, view=None):
    """Convenience wrapper for gluProject
    
    Automatically fills in the model, projection and viewing matrices
    if not provided.
    
    returns (winX,winY,winZ) doubles
    """
    if model is None:
        model = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
    if proj is None:
        proj = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
    if view is None:
        view = GL.glGetIntegerv(GL.GL_VIEWPORT)
    winX = _simple.GLdouble(0.0)
    winY = _simple.GLdouble(0.0)
    winZ = _simple.GLdouble(0.0)
    result = baseFunction(objX, objY, objZ, model, proj, view, winX, winY, winZ)
    if result is not None and result != _simple.GLU_TRUE:
        raise ValueError('Projection failed!')
    return (winX.value, winY.value, winZ.value)