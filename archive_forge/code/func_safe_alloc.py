import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
def safe_alloc(self, count):
    """Allocate vertices, resizing the buffers if necessary."""
    try:
        return self.allocator.alloc(count)
    except allocation.AllocatorMemoryException as e:
        capacity = _nearest_pow2(e.requested_capacity)
        for buffer, _ in self.buffer_attributes:
            buffer.resize(capacity * buffer.attribute_stride)
        self.allocator.set_capacity(capacity)
        return self.allocator.alloc(count)