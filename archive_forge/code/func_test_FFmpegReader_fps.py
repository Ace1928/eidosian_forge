import os
import sys
import numpy as np
from numpy.testing import assert_equal
import skvideo.datasets
import skvideo.io
@unittest.skipIf(not skvideo._HAS_FFMPEG, 'FFmpeg required for this test.')
def test_FFmpegReader_fps():
    reader = skvideo.io.FFmpegReader(skvideo.datasets.bigbuckbunny(), outputdict={'-r': '10'}, verbosity=0)
    T = 0
    M = 0
    N = 0
    C = 0
    accumulation = 0
    for frame in reader.nextFrame():
        M, N, C = frame.shape
        accumulation += np.sum(frame)
        T += 1
    assert_equal(T, 54)
    assert_equal(M, 720)
    assert_equal(N, 1280)
    assert_equal(C, 3)
    assert_equal(accumulation / (T * M * N * C), 109.16349342126415)