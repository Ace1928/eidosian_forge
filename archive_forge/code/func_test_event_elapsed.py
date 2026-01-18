import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_event_elapsed(self):
    N = 32
    dary = cuda.device_array(N, dtype=np.double)
    evtstart = cuda.event()
    evtend = cuda.event()
    evtstart.record()
    cuda.to_device(np.arange(N, dtype=np.double), to=dary)
    evtend.record()
    evtend.wait()
    evtend.synchronize()
    evtstart.elapsed_time(evtend)