import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def array_of_object(x):
    return x