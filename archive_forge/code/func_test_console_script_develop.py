import glob
import os
import sys
import tarfile
import fixtures
from pbr.tests import base
def test_console_script_develop(self):
    """Test that we develop a non-pkg-resources console script."""
    if sys.version_info < (3, 0):
        self.skipTest('Fails with recent virtualenv due to https://github.com/pypa/virtualenv/issues/1638')
    if os.name == 'nt':
        self.skipTest('Windows support is passthrough')
    self.useFixture(fixtures.EnvironmentVariable('PYTHONPATH', '.:%s' % self.temp_dir))
    stdout, _, return_code = self.run_setup('develop', '--install-dir=%s' % self.temp_dir)
    self.check_script_install(stdout)