import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_footprint_diamond(self):
    """Test diamond footprints"""
    self.strel_worker('data/diamond-matlab-output.npz', footprints.diamond)