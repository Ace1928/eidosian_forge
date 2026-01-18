import ctypes
import weakref
from OpenGL._bytes import long, integer_types
def noteObject(self, object):
    """Note object for later retrieval as a Python object pointer
        
        This is the registration point for "original object return", returns 
        a void pointer to the Python object, though this is, effectively, an 
        opaque value.
        """
    identity = id(object)
    try:
        self.dataPointers[identity] = object
    except AttributeError as err:
        self.dataPointers = {identity: object}
    return identity