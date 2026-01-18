import numpy as np
import unittest
from numba import jit, vectorize, int8, int16, int32
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (Color, Shape, Shake,
def getattr_usecase(a):
    return a is Color.red