from typing import Tuple
import numpy
import scipy.sparse
from .. import base_matrix_interface as base
def scalar_matrix(self, value, shape: Tuple[int, ...]):
    return numpy.zeros(shape, dtype='float64') + value