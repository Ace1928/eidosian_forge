import io
import os
import tempfile
from contextlib import contextmanager
from rpy2.robjects.packages import importr, WeakPackage
@contextmanager
def render_to_file(device, *device_args, **device_kwargs):
    """
    Context manager that returns a R figures in a file object.

    :param device: an R "device" function. This function is expected
                   to take a filename as its first argument.

    """
    current = dev_cur()[0]
    try:
        device(*device_args, **device_kwargs)
        yield None
    finally:
        if current != dev_cur()[0]:
            dev_off()