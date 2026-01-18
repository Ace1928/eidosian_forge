import math
import struct
from ctypes import create_string_buffer
def avgpp(cp, size):
    _check_params(len(cp), size)
    sample_count = _sample_count(cp, size)
    prevextremevalid = False
    prevextreme = None
    avg = 0
    nextreme = 0
    prevval = getsample(cp, size, 0)
    val = getsample(cp, size, 1)
    prevdiff = val - prevval
    for i in range(1, sample_count):
        val = getsample(cp, size, i)
        diff = val - prevval
        if diff * prevdiff < 0:
            if prevextremevalid:
                avg += abs(prevval - prevextreme)
                nextreme += 1
            prevextremevalid = True
            prevextreme = prevval
        prevval = val
        if diff != 0:
            prevdiff = diff
    if nextreme == 0:
        return 0
    return avg / nextreme