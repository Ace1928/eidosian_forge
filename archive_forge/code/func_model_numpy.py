from numba import jit
import unittest
@jit(error_model='numpy')
def model_numpy(val):
    return 1 / val