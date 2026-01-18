import os
import textwrap
import unittest
from gae_ext_runtime import testutil
def test_generate_with_php_files(self):
    self.write_file('index.php', 'index')
    self.generate_configs()
    app_yaml = self.file_contents('app.yaml')
    self.assertIn('runtime: php\n', app_yaml)
    self.assertIn('env: flex\n', app_yaml)
    self.assertIn('runtime_config:\n  document_root: .\n', app_yaml)
    self.assertNotIn('entrypoint', app_yaml)
    self.assertFalse(os.path.exists(self.full_path('Dockerfile')))
    self.assertFalse(os.path.exists(self.full_path('.dockerignore')))