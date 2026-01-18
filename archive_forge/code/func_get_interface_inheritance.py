import sys
import ctypes
from pyglet.util import debug_print
@classmethod
def get_interface_inheritance(cls):
    """Returns the types of all interfaces implemented by this interface, up to but not
        including the base `Interface`.
        `Interface` does not represent an actual interface, but merely the base concept of
        them, so viewing it as part of an interface's inheritance chain is meaningless.
        """
    return cls.__mro__[:cls.__mro__.index(Interface)]