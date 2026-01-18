import unittest
from numba import njit
from numba.core import types
def is_in_mandelbrot(c):
    i = 0
    z = 0j
    for i in range(100):
        z = z ** 2 + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return False
    return True