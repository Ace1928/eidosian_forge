import os
import sys
from ..transport.http import ca_bundle
from . import TestCaseInTempDir, TestSkipped
def test_in_path(self):
    if sys.platform != 'win32':
        raise TestSkipped('Searching in PATH implemented only for win32')
    os.mkdir('foo')
    in_dir = os.path.join(self.test_dir, 'foo')
    self._make_file(in_dir=in_dir)
    self.overrideEnv('PATH', in_dir)
    shouldbe = os.path.join(in_dir, 'curl-ca-bundle.crt')
    self.assertEqual(shouldbe, ca_bundle.get_ca_path(use_cache=False))