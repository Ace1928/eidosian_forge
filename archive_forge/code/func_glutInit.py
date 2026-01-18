from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
def glutInit(*args):
    """Initialise the GLUT library"""
    global INITIALIZED
    if INITIALIZED:
        return args
    INITIALIZED = True
    if args:
        arg, args = (args[0], args[1:])
        count = None
        if isinstance(arg, integer_types):
            count = arg
            if count != len(args):
                raise ValueError('Specified count of %s does not match length (%s) of argument list %s' % (count, len(args), args))
        elif isinstance(arg, (bytes, unicode)):
            args = (arg,) + args
            count = len(args)
        else:
            args = arg
            count = len(args)
    else:
        count = 0
        args = []
    args = [as_8_bit(x) for x in args]
    if not count:
        count, args = (1, [as_8_bit('foo')])
    holder = (ctypes.c_char_p * len(args))()
    for i, arg in enumerate(args):
        holder[i] = arg
    count = ctypes.c_int(count)
    import os
    currentDirectory = os.getcwd()
    try:
        _base_glutInit(ctypes.byref(count), holder)
    finally:
        os.chdir(currentDirectory)
    return [holder[i] for i in range(count.value)]