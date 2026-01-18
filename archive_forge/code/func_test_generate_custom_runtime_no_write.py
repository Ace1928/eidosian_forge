import os
import textwrap
import unittest
from gae_ext_runtime import testutil
def test_generate_custom_runtime_no_write(self):
    """Tests generate_config_data with custom runtime."""
    self.write_file('index.php', 'index')
    cfg_files = self.generate_config_data(custom=True)
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', self.preamble() + textwrap.dedent('            ENV DOCUMENT_ROOT /app\n            '))
    self.assert_genfile_exists_with_contents(cfg_files, '.dockerignore', self.license() + textwrap.dedent('            .dockerignore\n            Dockerfile\n            .git\n            .hg\n            .svn\n            '))