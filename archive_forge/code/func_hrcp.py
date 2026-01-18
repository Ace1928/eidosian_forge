from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
def hrcp(self, x):
    return np.reciprocal(x, dtype=np.float16)