import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def getExtensionProcedure(self):
    eglGetProcAddress = self.EGL.eglGetProcAddress
    eglGetProcAddress.restype = ctypes.c_void_p
    return eglGetProcAddress