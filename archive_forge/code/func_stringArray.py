import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
def stringArray(self, arg, baseOperation, args):
    """Create basic array-of-strings object from pyArg"""
    if isinstance(arg, (bytes, unicode)):
        arg = [arg]
    value = [as_8_bit(x) for x in arg]
    return value