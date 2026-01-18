import glob
import os
import sys
import tarfile
import fixtures
from pbr.tests import base
def test_console_script_install(self):
    """Test that we install a non-pkg-resources console script."""
    if os.name == 'nt':
        self.skipTest('Windows support is passthrough')
    stdout, _, return_code = self.run_setup('install_scripts', '--install-dir=%s' % self.temp_dir)
    self.useFixture(fixtures.EnvironmentVariable('PYTHONPATH', '.'))
    self.check_script_install(stdout)