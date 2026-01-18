import warnings
import cupy
from cupy_backends.cuda.api import runtime
import cupyx.scipy.signal._signaltools as filtering
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal.windows._windows import get_window
def detrend_func(d):
    d = cupy.rollaxis(d, -1, axis)
    d = detrend(d)
    return cupy.rollaxis(d, axis, len(d.shape))