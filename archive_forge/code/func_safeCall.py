from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
def safeCall(*args, **named):
    """Safe calling of GUI callbacks, exits on failures"""
    try:
        if not CurrentContextIsValid():
            raise RuntimeError('No valid context!')
        return function(*args, **named)
    except Exception as err:
        traceback.print_exc()
        sys.stderr.write('GLUT %s callback %s with %s,%s failed: returning None %s\n' % (self.typeName, function, args, named, err))
        os._exit(1)