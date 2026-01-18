import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_get_index_dtype(self):
    imax = np.int64(np.iinfo(np.int32).max)
    too_big = imax + 1
    a1 = np.ones(90, dtype='uint32')
    a2 = np.ones(90, dtype='uint32')
    assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)), np.dtype('int32'))
    a1[-1] = imax
    assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)), np.dtype('int32'))
    a1[-1] = too_big
    assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)), np.dtype('int64'))
    a1 = np.ones(89, dtype='uint32')
    a2 = np.ones(89, dtype='uint32')
    assert_equal(np.dtype(sputils.get_index_dtype((a1, a2))), np.dtype('int64'))
    a1 = np.ones(12, dtype='uint32')
    a2 = np.ones(12, dtype='uint32')
    assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), maxval=too_big, check_contents=True)), np.dtype('int64'))
    a1[-1] = too_big
    assert_equal(np.dtype(sputils.get_index_dtype((a1, a2), maxval=too_big)), np.dtype('int64'))