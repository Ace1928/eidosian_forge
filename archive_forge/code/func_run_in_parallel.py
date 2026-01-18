import os
import re
import struct
import threading
import functools
import inspect
from tempfile import NamedTemporaryFile
import numpy as np
from numpy import testing
from numpy.testing import (
from .. import data, io
from ..data._fetchers import _fetch
from ..util import img_as_uint, img_as_float, img_as_int, img_as_ubyte
from ._warnings import expected_warnings
import pytest
def run_in_parallel(num_threads=2, warnings_matching=None):
    """Decorator to run the same function multiple times in parallel.

    This decorator is useful to ensure that separate threads execute
    concurrently and correctly while releasing the GIL.

    Parameters
    ----------
    num_threads : int, optional
        The number of times the function is run in parallel.

    warnings_matching: list or None
        This parameter is passed on to `expected_warnings` so as not to have
        race conditions with the warnings filters. A single
        `expected_warnings` context manager is used for all threads.
        If None, then no warnings are checked.

    """
    assert num_threads > 0

    def wrapper(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):
            with expected_warnings(warnings_matching):
                threads = []
                for i in range(num_threads - 1):
                    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
                    threads.append(thread)
                for thread in threads:
                    thread.start()
                func(*args, **kwargs)
                for thread in threads:
                    thread.join()
        return inner
    return wrapper