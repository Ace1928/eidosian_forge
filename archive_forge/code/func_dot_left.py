from numpy.testing import assert_equal
import numpy as np
def dot_left(self, a):
    """ b = C a
        """
    return np.dot(self.transf_matrix, a)