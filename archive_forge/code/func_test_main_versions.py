from numpy.testing import assert_, assert_raises
from numpy.lib import NumpyVersion
def test_main_versions():
    assert_(NumpyVersion('1.8.0') == '1.8.0')
    for ver in ['1.9.0', '2.0.0', '1.8.1', '10.0.1']:
        assert_(NumpyVersion('1.8.0') < ver)
    for ver in ['1.7.0', '1.7.1', '0.9.9']:
        assert_(NumpyVersion('1.8.0') > ver)