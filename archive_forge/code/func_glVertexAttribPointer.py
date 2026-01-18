from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.VERSION.GLES2_2_0 import *
from OpenGL.raw.GLES2.VERSION.GLES2_2_0 import _EXTENSION_NAME
from OpenGL import converters
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL._bytes import _NULL_8_BYTE
from OpenGL import contextdata 
@_lazy(glVertexAttribPointer)
def glVertexAttribPointer(baseOperation, index, size, type, normalized, stride, pointer):
    """Set an attribute pointer for a given shader (index)

    index -- the index of the generic vertex to bind, see
        glGetAttribLocation for retrieval of the value,
        note that index is a global variable, not per-shader
    size -- number of basic elements per record, 1,2,3, or 4
    type -- enum constant for data-type
    normalized -- whether to perform int to float
        normalization on integer-type values
    stride -- stride in machine units (bytes) between
        consecutive records, normally used to create
        "interleaved" arrays
    pointer -- data-pointer which provides the data-values,
        normally a vertex-buffer-object or offset into the
        same.

    This implementation stores a copy of the data-pointer
    in the contextdata structure in order to prevent null-
    reference errors in the renderer.
    """
    array = arrays.ArrayDatatype.asArray(pointer, type)
    key = ('vertex-attrib', index)
    contextdata.setValue(key, array)
    return baseOperation(index, size, type, normalized, stride, arrays.ArrayDatatype.voidDataPointer(array))