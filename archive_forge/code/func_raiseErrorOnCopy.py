import ctypes, _ctypes
from OpenGL.raw.GL import _types 
from OpenGL.arrays import _arrayconstants as GL_1_1
from OpenGL import constant, error
from OpenGL._configflags import ERROR_ON_COPY
from OpenGL.arrays import formathandler
from OpenGL._bytes import bytes,unicode,as_8_bit
import operator
def raiseErrorOnCopy(self, value, *args, **named):
    raise error.CopyError('%s passed, cannot copy with ERROR_ON_COPY set, please use an array type which has native data-pointer support (e.g. numpy or ctypes arrays)' % (value.__class__.__name__,))