import os
import sys
import inspect
import contextlib
import numpy as np
import logging
from io import StringIO
import unittest
from numba.tests.support import SerialMixin, create_temp_module
from numba.core import dispatcher
from numba import jit_module
import numpy as np
from numba import jit, jit_module
@contextlib.contextmanager
def captured_logs(l):
    try:
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        l.addHandler(handler)
        yield buffer
    finally:
        l.removeHandler(handler)