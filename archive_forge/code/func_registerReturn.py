import ctypes
from OpenGL import plugins
def registerReturn(self):
    """Register this handler as the default return-type handler"""
    from OpenGL.arrays.arraydatatype import ArrayDatatype
    ArrayDatatype.getRegistry().registerReturn(self)