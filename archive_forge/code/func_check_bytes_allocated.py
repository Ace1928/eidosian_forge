import ctypes
from functools import wraps
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
def check_bytes_allocated(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        allocated_bytes = pa.total_allocated_bytes()
        try:
            return f(*args, **kwargs)
        finally:
            assert pa.total_allocated_bytes() == allocated_bytes
    return wrapper