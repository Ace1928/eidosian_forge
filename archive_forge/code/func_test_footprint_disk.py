import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_footprint_disk(self):
    """Test disk footprints"""
    self.strel_worker('data/disk-matlab-output.npz', footprints.disk)