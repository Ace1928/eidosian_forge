import weakref, ctypes, logging, os, glob
from OpenGL.platform import ctypesloader
from OpenGL import _opaque
def enumerate_devices():
    """Enumerate the set of gbm renderD* devices on the system
    
    Attempts to filter out any nvidia drivers with filter_bad_drivers
    along the way.
    """
    import glob
    return filter_bad_drivers(sorted(glob.glob('/dev/dri/card*')))