import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_fv_wrong_descriptor_types():
    """
    Test that DescriptorException is raised when the incorrect type for the
    descriptors is passed into the fisher_vector function.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print('scikit-learn is not installed. Please ensure it is installed in order to use the Fisher vector functionality.')
    with pytest.raises(DescriptorException):
        fisher_vector([[1, 2, 3, 4]], GaussianMixture())