from numpy.testing import assert_equal
import numpy as np
def r_nointer(self):
    """contrast/restriction matrix for no interaction
        """
    nia = self.n_interaction
    R_nointer = np.hstack((np.zeros((nia, self.nvars - nia)), np.eye(nia)))
    R_nointer_transf = self.transform.inv_dot_right(R_nointer)
    self.R_nointer_transf = R_nointer_transf
    return R_nointer_transf