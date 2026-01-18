from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.GLuint, _cs.GLsizei, arrays.GLuintArray, arrays.GLuint64Array, _cs.GLuint, _cs.GLbitfield, _cs.GLuint, _cs.GLuint, _cs.GLintptr, _cs.GLintptr, _cs.GLsizeiptr, _cs.GLsizei, arrays.GLuintArray, arrays.GLuint64Array)
def glAsyncCopyBufferSubDataNVX(waitSemaphoreCount, waitSemaphoreArray, fenceValueArray, readGpu, writeGpuMask, readBuffer, writeBuffer, readOffset, writeOffset, size, signalSemaphoreCount, signalSemaphoreArray, signalValueArray):
    pass