import numpy as np
import unittest
from numba import jit, vectorize, int8, int16, int32
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (Color, Shape, Shake,
def int_coerce_usecase(x):
    if x > RequestError.internal_error:
        return x - RequestError.not_found
    else:
        return x + Shape.circle