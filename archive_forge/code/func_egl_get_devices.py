from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.EXT.device_base import *
from OpenGL.raw.EGL.EXT.device_base import _EXTENSION_NAME
def egl_get_devices(max_count=10):
    """Utility function that retrieves platform devices"""
    devices = (_types.EGLDeviceEXT * max_count)()
    count = _types.EGLint()
    if eglQueryDevicesEXT(max_count, devices, count):
        return devices[:count.value]
    return []