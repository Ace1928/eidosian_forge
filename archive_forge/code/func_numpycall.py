import numpy as np
import numba as nb
from numpy.random import PCG64
from timeit import timeit
def numpycall():
    return rg.normal(size=n)