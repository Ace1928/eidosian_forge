import os
import sys
import numpy as np
from numpy.testing import assert_equal, assert_array_less
import skvideo
import skvideo.datasets
import skvideo.io
@unittest.skipIf(not skvideo._HAS_AVCONV, 'LibAV required for this test.')
def test_LibAVReader_16bits():
    reader16 = skvideo.io.LibAVReader(skvideo.datasets.bigbuckbunny(), outputdict={'-pix_fmt': 'rgb48le'}, verbosity=0)
    reader8 = skvideo.io.LibAVReader(skvideo.datasets.bigbuckbunny(), outputdict={'-pix_fmt': 'rgb24'}, verbosity=0)
    T = 0
    M = 0
    N = 0
    C = 0
    accumulation = 0
    for frame8, frame16 in zip(reader8.nextFrame(), reader16.nextFrame()):
        assert np.max(np.abs(frame8.astype('int32') - (frame16 // 256).astype('int32'))) < 4
        assert np.mean(np.abs(frame8.astype('float32') - (frame16 // 256).astype('float32'))) < 1.0
        M, N, C = frame8.shape
        accumulation += np.sum(frame16 // 256)
        T += 1
    assert_equal(T, 132)
    assert_equal(M, 720)
    assert_equal(N, 1280)
    assert_equal(C, 3)
    assert_equal(accumulation / (T * M * N * C), 108.89236060967751)