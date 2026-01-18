import numpy as np
import scipy.sparse as sp
from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface
@NDArrayInterface.scalar_const
def const_to_matrix(self, value, convert_scalars: bool=False):
    """Convert an arbitrary value into a matrix of type self.target_matrix.

        Args:
            value: The constant to be converted.
            convert_scalars: Should scalars be converted?

        Returns:
            A matrix of type self.target_matrix or a scalar.
        """
    if isinstance(value, list):
        return sp.csc_matrix(value, dtype=np.double).T
    if value.dtype in [np.double, complex]:
        dtype = value.dtype
    else:
        dtype = np.double
    return sp.csc_matrix(value, dtype=dtype)