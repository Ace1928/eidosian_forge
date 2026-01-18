import weakref, ctypes, logging, os, glob
from OpenGL.platform import ctypesloader
from OpenGL import _opaque
def close_device(device):
    """Close an opened gbm device"""
    gbm.gbm_device_destroy(device)
    try:
        handle = _DEVICE_HANDLES.pop(device)
        handle.close()
    except KeyError:
        pass