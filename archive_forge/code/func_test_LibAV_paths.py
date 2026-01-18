import warnings
from numpy.testing import assert_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
@unittest.skipIf(not skvideo._HAS_AVCONV, 'LibAV required for this test.')
def test_LibAV_paths():
    current_path = skvideo.getLibAVPath()
    current_version = skvideo.getLibAVVersion()
    assert current_version != '0.0', 'LibAV version not parsed.'
    skvideo.setLibAVPath('/')
    assert skvideo.getLibAVVersion() == '0.0', 'LibAV version is not zeroed out properly.'
    assert current_path != skvideo.getLibAVPath(), 'LibAV path did not update correctly'
    skvideo.setLibAVPath(current_path)
    assert current_path == skvideo.getLibAVPath(), 'LibAV path did not update correctly'
    assert skvideo.getLibAVVersion() == current_version, 'LibAV version is not loaded properly from valid FFmpeg.'