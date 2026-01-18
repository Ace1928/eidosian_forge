import asyncio
import functools
import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_add_callback_with_default_arg(self):
    callback_event = threading.Event()

    def callback(stream, status, arg):
        self.assertIsNone(arg)
        callback_event.set()
    stream = cuda.stream()
    stream.add_callback(callback)
    self.assertTrue(callback_event.wait(1.0))