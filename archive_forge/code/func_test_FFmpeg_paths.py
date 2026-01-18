import warnings
from numpy.testing import assert_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
@unittest.skipIf(not skvideo._HAS_FFMPEG, 'FFmpeg required for this test.')
def test_FFmpeg_paths():
    current_path = skvideo.getFFmpegPath()
    current_version = skvideo.getFFmpegVersion()
    assert current_version != '0.0.0', 'FFmpeg version not parsed.'
    skvideo.setFFmpegPath('/')
    assert skvideo.getFFmpegVersion() == '0.0.0', 'FFmpeg version is not zeroed out properly.'
    assert current_path != skvideo.getFFmpegPath(), 'FFmpeg path did not update correctly'
    skvideo.setFFmpegPath(current_path)
    assert current_path == skvideo.getFFmpegPath(), 'FFmpeg path did not update correctly'
    assert skvideo.getFFmpegVersion() == current_version, 'FFmpeg version is not loaded properly from valid FFmpeg.'