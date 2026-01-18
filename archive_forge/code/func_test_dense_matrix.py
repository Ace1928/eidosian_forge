import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import (TestCase, assert_array_almost_equal,
def test_dense_matrix(self):
    A = np.array([[1, 2, 3, 4, 0, 5, 0, 7], [0, 8, 7, 0, 1, 5, 9, 0], [1, 0, 0, 0, 0, 1, 2, 3]])
    test_vectors = ([-1.98931144, -1.56363389, -0.84115584, 2.2864762, 5.599141, 0.09286976, 1.37040802, -0.28145812], [697.92794044, -4091.65114008, -3327.42316335, 836.86906951, 99434.98929065, -1285.37653682, -4109.21503806, 2935.29289083])
    test_expected_orth = (0, 0)
    for i in range(len(test_vectors)):
        x = test_vectors[i]
        orth = test_expected_orth[i]
        assert_array_almost_equal(orthogonality(A, x), orth)