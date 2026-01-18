import numpy as np
import scipy.sparse as sp
from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface
def scalar_value(self, matrix):
    """Get the value of the passed matrix, interpreted as a scalar.
        """
    return matrix[0, 0]