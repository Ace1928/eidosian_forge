from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays.formathandler import FormatHandler
from OpenGL.raw.GL import _types 
from OpenGL import error
from OpenGL._bytes import bytes,unicode,as_8_bit
import ctypes,logging
from OpenGL._bytes import long, integer_types
import weakref
from OpenGL import acceleratesupport
def doBufferDeletion(*args, **named):
    while buffers:
        try:
            buffer = buffers.pop()
        except IndexError as err:
            break
        else:
            try:
                buf = gluint(buffer)
                self.glDeleteBuffers(1, buf)
            except (AttributeError, nfe, TypeError) as err:
                pass
    try:
        self._DELETERS_.pop(key)
    except KeyError as err:
        pass