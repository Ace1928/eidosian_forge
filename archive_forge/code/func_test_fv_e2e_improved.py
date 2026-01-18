import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_fv_e2e_improved():
    """
    Test the improved Fisher vector computation given a GMM returned from the
    learn_gmm function. We simply assert that the dimensionality of the
    resulting Fisher vector is correct.

    The dimensionality of a Fisher vector is given by 2KD + K, where K is the
    number of Gaussians specified in the associated GMM, and D is the
    dimensionality of the descriptors using to estimate the GMM.
    """
    dim = 128
    num_modes = 8
    expected_dim = 2 * num_modes * dim + num_modes
    descriptors = [np.random.random((np.random.randint(5, 30), dim)) for _ in range(10)]
    gmm = learn_gmm(descriptors, n_modes=num_modes)
    fisher_vec = fisher_vector(descriptors[0], gmm, improved=True)
    assert len(fisher_vec) == expected_dim