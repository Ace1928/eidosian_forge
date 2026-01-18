import unittest
import numpy as np
def mat_to_list(self, mat):
    """Convert a numpy matrix to a list.
        """
    if isinstance(mat, (np.matrix, np.ndarray)):
        return np.asarray(mat).flatten('F').tolist()
    else:
        return mat