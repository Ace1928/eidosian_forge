from numpy.testing import assert_equal
import numpy as np
def test_dummy_1d(self):
    x = np.array(['F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M'], dtype='|S1')
    d, labels = (np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]]), ['gender_F', 'gender_M'])
    res_d, res_labels = dummy_1d(x, varname='gender')
    assert_equal(res_d, d)
    assert_equal(res_labels, labels)