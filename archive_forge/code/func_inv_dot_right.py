from numpy.testing import assert_equal
import numpy as np
def inv_dot_right(self, z):
    """ x = z C^{-1}
        """
    return np.dot(z, self.invtransf_matrix)