from contextlib import contextmanager
import numpy as np
from numba import cuda
from numba.cuda.testing import (unittest, skip_on_cudasim,
from numba.tests.support import captured_stderr
from numba.core import config
@contextmanager
def check_ignored_exception(self, ctx):
    with captured_stderr() as cap:
        yield
        ctx.deallocations.clear()
    self.assertFalse(cap.getvalue())