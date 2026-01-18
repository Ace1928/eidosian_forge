import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_wrong_descriptor_format_2():
    """Test that DescriptorException is raised when descriptors are of
    different dimensionality.
    """
    with pytest.raises(DescriptorException):
        learn_gmm([np.zeros((5, 11)), np.zeros((4, 10))], n_modes=1)