from OpenGL import arrays
from OpenGL.arrays.arraydatatype import GLfloatArray
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL.GL.VERSION import GL_1_1 as full
from OpenGL.raw.GL import _errors
from OpenGL._bytes import bytes
from OpenGL import _configflags
from OpenGL._null import NULL as _NULL
import ctypes
def glColor(*args):
    """glColor*f* -- convenience function to dispatch on argument type

    dispatches to glColor3f, glColor2f, glColor4f, glColor3f, glColor2f, glColor4f
    depending on the arguments passed...
    """
    arglen = len(args)
    if arglen == 1:
        arg = arrays.GLfloatArray.asArray(args[0])
        function = glColorDispatch[arrays.GLfloatArray.arraySize(arg)]
        return function(arg)
    elif arglen == 2:
        return full.glColor2d(*args)
    elif arglen == 3:
        return full.glColor3d(*args)
    elif arglen == 4:
        return full.glColor4d(*args)
    else:
        raise ValueError("Don't know how to handle arguments: %s" % (args,))