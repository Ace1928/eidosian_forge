import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_wrong_descriptor_format_3():
    """Test that DescriptorException is raised when not all descriptors are of
    rank 2.
    """
    with pytest.raises(DescriptorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10, 1))], n_modes=1)