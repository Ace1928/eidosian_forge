import unittest
import numpy as np
import numba
@numba.njit
def do_stuff(gen):
    return gen.random(size=int(size / 2))