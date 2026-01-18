import sys
from numba import cuda, njit
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.cudapy.cache_usecases import CUDAUseCase, UseCase
def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)