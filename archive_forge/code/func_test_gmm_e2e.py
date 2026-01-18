import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_e2e():
    """
    Test the GMM estimation. Since this is essentially a wrapper for the
    scikit-learn GaussianMixture class, the testing of the actual inner
    workings of the GMM estimation is left to scikit-learn and its
    dependencies.

    We instead simply assert that the estimation was successful based on the
    fact that the GMM object will have associated mixture weights, means, and
    variances after estimation is successful/complete.
    """
    gmm = learn_gmm(np.random.random((100, 64)), n_modes=5)
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.weights_ is not None