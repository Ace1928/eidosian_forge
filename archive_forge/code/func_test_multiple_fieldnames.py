from os.path import dirname, join as pjoin
from numpy.testing import assert_
from pytest import raises as assert_raises
from scipy.io.matlab._mio import loadmat
def test_multiple_fieldnames():
    multi_fname = pjoin(TEST_DATA_PATH, 'nasty_duplicate_fieldnames.mat')
    vars = loadmat(multi_fname)
    funny_names = vars['Summary'].dtype.names
    assert_({'_1_Station_Q', '_2_Station_Q', '_3_Station_Q'}.issubset(funny_names))