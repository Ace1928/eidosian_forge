import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
def requires_memory(free_bytes):
    """Decorator to skip a test if not enough memory is available"""
    import pytest

    def decorator(func):

        @wraps(func)
        def wrapper(*a, **kw):
            msg = check_free_memory(free_bytes)
            if msg is not None:
                pytest.skip(msg)
            try:
                return func(*a, **kw)
            except MemoryError:
                pytest.xfail('MemoryError raised')
        return wrapper
    return decorator