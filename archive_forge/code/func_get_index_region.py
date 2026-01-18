import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
def get_index_region(self, start, count):
    """Get a data from a region of the index buffer.

        :Parameters:
            `start` : int
                Start of the region to map.
            `count` : int
                Number of indices to map.

        :rtype: Array of int
        """
    byte_start = self.index_element_size * start
    byte_count = self.index_element_size * count
    ptr_type = ctypes.POINTER(self.index_c_type * count)
    map_ptr = self.index_buffer.map_range(byte_start, byte_count, ptr_type)
    data = map_ptr[:]
    self.index_buffer.unmap()
    return data