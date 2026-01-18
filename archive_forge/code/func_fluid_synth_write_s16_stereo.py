from ctypes import *
from ctypes.util import find_library
import os
def fluid_synth_write_s16_stereo(synth, len):
    """Return generated samples in stereo 16-bit format

    Return value is a Numpy array of samples.

    """
    import numpy
    buf = create_string_buffer(len * 4)
    fluid_synth_write_s16(synth, len, buf, 0, 2, buf, 1, 2)
    return numpy.frombuffer(buf[:], dtype=numpy.int16)