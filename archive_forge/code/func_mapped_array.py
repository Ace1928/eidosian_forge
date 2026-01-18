from contextlib import contextmanager
import numpy as np
from_record_like = None
def mapped_array(*args, **kwargs):
    for unused_arg in ('portable', 'wc'):
        if unused_arg in kwargs:
            kwargs.pop(unused_arg)
    return device_array(*args, **kwargs)