from OpenGL import platform, error, wrapper, contextdata, converters, constant
from OpenGL.arrays import arrayhelpers, arraydatatype
from OpenGL.raw.GL.VERSION import GL_1_1 as _simple
import ctypes
def glDrawElementsTyped(type, suffix):
    arrayType = arraydatatype.GL_CONSTANT_TO_ARRAY_TYPE[type]
    function = wrapper.wrapper(_simple.glDrawElements).setPyConverter('type').setCConverter('type', type).setPyConverter('count').setCConverter('count', arrayhelpers.AsArrayTypedSize('indices', arrayType)).setPyConverter('indices', arrayhelpers.AsArrayTyped('indices', arrayType)).setReturnValues(wrapper.returnPyArgument('indices'))
    return function