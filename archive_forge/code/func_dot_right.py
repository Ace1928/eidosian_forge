from numpy.testing import assert_equal
import numpy as np
def dot_right(self, x):
    """ z = x C
        """
    return np.dot(x, self.transf_matrix)