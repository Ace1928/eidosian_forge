import threading
from numba import cuda
from numba.cuda.cudadrv.driver import driver
from numba.cuda.testing import unittest, ContextResettingTestCase
from queue import Queue
def newthread(exception_queue):
    try:
        devices = range(driver.get_device_count())
        for _ in range(2):
            for d in devices:
                cuda.select_device(d)
                cuda.close()
    except Exception as e:
        exception_queue.put(e)