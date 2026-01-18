from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
@property
def laneid(self):
    return threading.current_thread().thread_id % 32