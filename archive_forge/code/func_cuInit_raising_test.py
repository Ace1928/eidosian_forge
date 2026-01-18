import multiprocessing as mp
import os
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.cudadrv.error import CudaSupportError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def cuInit_raising_test(result_queue):
    driver.cuInit = cuInit_raising
    success = False
    msg = None
    try:
        cuda.device_array(1)
    except CudaSupportError as e:
        success = True
        msg = e.msg
    result_queue.put((success, msg))