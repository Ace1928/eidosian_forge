from os.path import join, sep, dirname
from numpy.distutils.misc_util import (
from numpy.testing import (
def test_get_shared_lib_extension(self):
    import sys
    ext = get_shared_lib_extension(is_python_ext=False)
    if sys.platform.startswith('linux'):
        assert_equal(ext, '.so')
    elif sys.platform.startswith('gnukfreebsd'):
        assert_equal(ext, '.so')
    elif sys.platform.startswith('darwin'):
        assert_equal(ext, '.dylib')
    elif sys.platform.startswith('win'):
        assert_equal(ext, '.dll')
    assert_(get_shared_lib_extension(is_python_ext=True))