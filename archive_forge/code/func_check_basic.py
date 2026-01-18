import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO
def check_basic(self, *kws):
    s, o = exec_command.exec_command('"%s" -c "raise \'Ignore me.\'"' % self.pyexe, **kws)
    assert_(s != 0)
    assert_(o != '')
    s, o = exec_command.exec_command('"%s" -c "import sys;sys.stderr.write(\'0\');sys.stderr.write(\'1\');sys.stderr.write(\'2\')"' % self.pyexe, **kws)
    assert_(s == 0)
    assert_(o == '012')
    s, o = exec_command.exec_command('"%s" -c "import sys;sys.exit(15)"' % self.pyexe, **kws)
    assert_(s == 15)
    assert_(o == '')
    s, o = exec_command.exec_command('"%s" -c "print(\'Heipa\'")' % self.pyexe, **kws)
    assert_(s == 0)
    assert_(o == 'Heipa')