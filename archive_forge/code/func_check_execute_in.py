import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO
def check_execute_in(self, **kws):
    with tempdir() as tmpdir:
        fn = 'file'
        tmpfile = os.path.join(tmpdir, fn)
        with open(tmpfile, 'w') as f:
            f.write('Hello')
        s, o = exec_command.exec_command('"%s" -c "f = open(\'%s\', \'r\'); f.close()"' % (self.pyexe, fn), **kws)
        assert_(s != 0)
        assert_(o != '')
        s, o = exec_command.exec_command('"%s" -c "f = open(\'%s\', \'r\'); print(f.read()); f.close()"' % (self.pyexe, fn), execute_in=tmpdir, **kws)
        assert_(s == 0)
        assert_(o == 'Hello')