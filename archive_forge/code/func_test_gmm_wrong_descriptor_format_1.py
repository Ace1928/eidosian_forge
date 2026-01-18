import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_wrong_descriptor_format_1():
    """Test that DescriptorException is raised when wrong type for descriptions
    is passed.
    """
    with pytest.raises(DescriptorException):
        learn_gmm('completely wrong test', n_modes=1)