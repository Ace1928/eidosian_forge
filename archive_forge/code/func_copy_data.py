from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays.formathandler import FormatHandler
from OpenGL.raw.GL import _types 
from OpenGL import error
from OpenGL._bytes import bytes,unicode,as_8_bit
import ctypes,logging
from OpenGL._bytes import long, integer_types
import weakref
from OpenGL import acceleratesupport
def copy_data(self):
    """Copy our data into the buffer on the GL side (if required)
            
            Ensures that the GL's version of the data in the VBO matches our 
            internal view of the data, either by copying the entire data-set 
            over with glBufferData or by updating the already-transferred 
            data with glBufferSubData.
            """
    assert self.buffers, 'Should do create_buffers before copy_data'
    if self.copied:
        if self._copy_segments:
            while self._copy_segments:
                start, size, data = self._copy_segments.pop(0)
                dataptr = ArrayDatatype.voidDataPointer(data)
                self.implementation.glBufferSubData(self.target, start, size, dataptr)
    else:
        if self.data is not None and self.size is None:
            self.size = ArrayDatatype.arrayByteCount(self.data)
        self.implementation.glBufferData(self.target, self.size, self.data, self.usage)
        self.copied = True