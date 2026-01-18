from os.path import join, sep, dirname
from numpy.distutils.misc_util import (
from numpy.testing import (
def test_gpaths(self):
    local_path = minrelpath(join(dirname(__file__), '..'))
    ls = gpaths('command/*.py', local_path)
    assert_(join(local_path, 'command', 'build_src.py') in ls, repr(ls))
    f = gpaths('system_info.py', local_path)
    assert_(join(local_path, 'system_info.py') == f[0], repr(f))