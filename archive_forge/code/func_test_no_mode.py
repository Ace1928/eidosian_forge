import os
import stat
import sys
from .. import atomicfile, osutils
from . import TestCaseInTempDir, TestSkipped
def test_no_mode(self):
    umask = osutils.get_umask()
    f = atomicfile.AtomicFile('test', mode='wb')
    f.write(b'foo\n')
    f.commit()
    st = os.lstat('test')
    self.assertEqualMode(438 & ~umask, stat.S_IMODE(st.st_mode))