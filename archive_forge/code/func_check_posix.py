import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO
def check_posix(self, **kws):
    s, o = exec_command.exec_command('echo Hello', **kws)
    assert_(s == 0)
    assert_(o == 'Hello')
    s, o = exec_command.exec_command('echo $AAA', **kws)
    assert_(s == 0)
    assert_(o == '')
    s, o = exec_command.exec_command('echo "$AAA"', AAA='Tere', **kws)
    assert_(s == 0)
    assert_(o == 'Tere')
    s, o = exec_command.exec_command('echo "$AAA"', **kws)
    assert_(s == 0)
    assert_(o == '')
    if 'BBB' not in os.environ:
        os.environ['BBB'] = 'Hi'
        s, o = exec_command.exec_command('echo "$BBB"', **kws)
        assert_(s == 0)
        assert_(o == 'Hi')
        s, o = exec_command.exec_command('echo "$BBB"', BBB='Hey', **kws)
        assert_(s == 0)
        assert_(o == 'Hey')
        s, o = exec_command.exec_command('echo "$BBB"', **kws)
        assert_(s == 0)
        assert_(o == 'Hi')
        del os.environ['BBB']
        s, o = exec_command.exec_command('echo "$BBB"', **kws)
        assert_(s == 0)
        assert_(o == '')
    s, o = exec_command.exec_command('this_is_not_a_command', **kws)
    assert_(s != 0)
    assert_(o != '')
    s, o = exec_command.exec_command('echo path=$PATH', **kws)
    assert_(s == 0)
    assert_(o != '')
    s, o = exec_command.exec_command('"%s" -c "import sys,os;sys.stderr.write(os.name)"' % self.pyexe, **kws)
    assert_(s == 0)
    assert_(o == 'posix')