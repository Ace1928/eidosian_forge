from ctypes import *
from ctypes.util import find_library
import os
def raw_audio_string(data):
    """Return a string of bytes to send to soundcard

    Input is a numpy array of samples.  Default output format
    is 16-bit signed (other formats not currently supported).

    """
    import numpy
    return data.astype(numpy.int16).tostring()