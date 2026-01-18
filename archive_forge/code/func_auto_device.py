from contextlib import contextmanager
import numpy as np
from_record_like = None
def auto_device(ary, stream=0, copy=True):
    if isinstance(ary, FakeCUDAArray):
        return (ary, False)
    if not isinstance(ary, np.void):
        ary = np.array(ary, copy=False, subok=True)
    return (to_device(ary, stream, copy), True)