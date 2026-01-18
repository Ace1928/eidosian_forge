import os
import sys
import numpy as np
from numpy.testing import assert_equal, assert_array_less
import skvideo
import skvideo.datasets
import skvideo.io
@unittest.skipIf(not skvideo._HAS_AVCONV, 'LibAV required for this test.')
def test_LibAVReader_aboveversion9():
    if not skvideo._HAS_AVCONV:
        return 0
    if np.int(skvideo._LIBAV_MAJOR_VERSION) < 9:
        return 0
    reader = skvideo.io.LibAVReader(skvideo.datasets.bigbuckbunny(), verbosity=0)
    T = 0
    M = 0
    N = 0
    C = 0
    accumulation = 0
    for frame in reader.nextFrame():
        M, N, C = frame.shape
        accumulation += np.sum(frame)
        T += 1
    assert_equal(T, 132)
    assert_equal(M, 720)
    assert_equal(N, 1280)
    assert_equal(C, 3)
    assert_equal(accumulation / (T * M * N * C), 109.28332841215979)
    reader = skvideo.io.LibAVReader(skvideo.datasets.bigbuckbunny(), verbosity=0)
    T = 0
    M = 0
    N = 0
    C = 0
    accumulation = 0
    for frame in reader:
        M, N, C = frame.shape
        accumulation += np.sum(frame)
        T += 1
    assert_equal(T, 132)
    assert_equal(M, 720)
    assert_equal(N, 1280)
    assert_equal(C, 3)
    assert_equal(accumulation / (T * M * N * C), 109.28332841215979)