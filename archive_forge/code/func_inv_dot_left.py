from numpy.testing import assert_equal
import numpy as np
def inv_dot_left(self, b):
    """ a = C^{-1} b
        """
    return np.dot(self.invtransf_matrix, b)