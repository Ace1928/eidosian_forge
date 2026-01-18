from numpy.testing import assert_equal
import numpy as np
def ftest_interaction(self):
    """ttests for no-interaction terms are zero
        """
    R_nointer_transf = self.r_nointer()
    return self.resols.f_test(R_nointer_transf)