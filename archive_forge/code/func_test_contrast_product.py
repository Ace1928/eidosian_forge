from numpy.testing import assert_equal
import numpy as np
def test_contrast_product(self):
    res_cp = contrast_product(self.v1name, self.v2name)
    res_t = [0] * 6
    res_t[0] = ['a0_b0', 'a0_b1', 'a1_b0', 'a1_b1', 'a2_b0', 'a2_b1']
    res_t[1] = np.array([[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 1.0]])
    res_t[2] = ['a1_b0-a0_b0', 'a1_b1-a0_b1', 'a2_b0-a0_b0', 'a2_b1-a0_b1']
    res_t[3] = np.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
    res_t[4] = ['a0_b1-a0_b0', 'a1_b1-a1_b0', 'a2_b1-a2_b0']
    for ii in range(5):
        np.testing.assert_equal(res_cp[ii], res_t[ii], err_msg=str(ii))