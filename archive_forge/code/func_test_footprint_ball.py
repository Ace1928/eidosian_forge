import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_footprint_ball(self):
    """Test ball footprints"""
    self.strel_worker_3d('data/disk-matlab-output.npz', footprints.ball)