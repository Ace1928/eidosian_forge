import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO
def check_nt(self, **kws):
    s, o = exec_command.exec_command('cmd /C echo path=%path%')
    assert_(s == 0)
    assert_(o != '')
    s, o = exec_command.exec_command('"%s" -c "import sys;sys.stderr.write(sys.platform)"' % self.pyexe)
    assert_(s == 0)
    assert_(o == 'win32')