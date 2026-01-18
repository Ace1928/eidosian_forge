import multiprocessing as mp
import logging
import traceback
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import (skip_on_cudasim, skip_with_cuda_python,
from numba.tests.support import linux_only
def child_test():
    from numba import cuda, int32, void
    from numba.core import config
    import io
    import numpy as np
    import threading
    config.CUDA_PER_THREAD_DEFAULT_STREAM = 1
    logbuf = io.StringIO()
    handler = logging.StreamHandler(logbuf)
    cudadrv_logger = logging.getLogger('numba.cuda.cudadrv.driver')
    cudadrv_logger.addHandler(handler)
    cudadrv_logger.setLevel(logging.DEBUG)
    N = 2 ** 16
    N_THREADS = 10
    N_ADDITIONS = 4096
    np.random.seed(1)
    x = np.random.randint(low=0, high=1000, size=N, dtype=np.int32)
    r = np.zeros_like(x)
    xs = [cuda.to_device(x) for _ in range(N_THREADS)]
    rs = [cuda.to_device(r) for _ in range(N_THREADS)]
    n_threads = 256
    n_blocks = N // n_threads
    stream = cuda.default_stream()

    @cuda.jit(void(int32[::1], int32[::1]))
    def f(r, x):
        i = cuda.grid(1)
        if i > len(r):
            return
        for j in range(N_ADDITIONS):
            r[i] += x[i]

    def kernel_thread(n):
        f[n_blocks, n_threads, stream](rs[n], xs[n])
    threads = [threading.Thread(target=kernel_thread, args=(i,)) for i in range(N_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    cuda.synchronize()
    expected = x * N_ADDITIONS
    for i in range(N_THREADS):
        np.testing.assert_equal(rs[i].copy_to_host(), expected)
    handler.flush()
    return logbuf.getvalue()