import numpy
import numpy.random
from .py import gemm, einsum
from timeit import default_timer as timer
def get_numpy_blas():
    blas_libs = numpy.__config__.blas_opt_info['libraries']
    return blas_libs[0]