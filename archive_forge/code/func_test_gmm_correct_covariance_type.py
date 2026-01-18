import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_correct_covariance_type():
    """Test that GMM estimation is successful when the correct covariance type
    is passed in as a keyword argument.
    """
    gmm = learn_gmm(np.random.random((10, 10)), n_modes=2, gm_args={'covariance_type': 'diag'})
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.weights_ is not None