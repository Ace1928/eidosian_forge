from contextlib import contextmanager
import numpy as np
from_record_like = None
def device_array(*args, **kwargs):
    stream = kwargs.pop('stream') if 'stream' in kwargs else 0
    return FakeCUDAArray(np.ndarray(*args, **kwargs), stream=stream)